"""解码、EXIF 方向落实、ICC 到 sRGB 的标准化。

设计约束：
- 原图只读，Pillow 以只读方式打开，不写回。
- 方向必须先落实到像素，再做尺寸计算与统计。
- 含 Alpha 的图片首版跳过（不自行选择背景色合成）。
- 多帧/动图按错误处理，不做首帧猜测。
- 无 ICC 时按 sRGB 解释并给出警告；ICC 损坏或转换失败必须报错，不得静默换空间。
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .errors import ErrorCode, PreprocessError, SkipImage, SkipReason, Stage, WarningCode
from .util import round_float

SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF", "WEBP"}
_GRAY_MODES = {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}
_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}

_SRGB_PROFILE = None
_SRGB_ICC_BYTES: bytes | None = None
_SRGB_ICC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "srgb.icc")


def srgb_icc_bytes() -> bytes:
    """内置 sRGB ICC 字节（写入预览）。

    优先使用随包分发的固定 ICC 文件，保证不同 lcms/Pillow 版本下预览字节稳定；
    文件缺失时才回退到 Pillow 生成（会让 preview_sha256 依赖 lcms 版本）。
    """
    global _SRGB_PROFILE, _SRGB_ICC_BYTES
    if _SRGB_ICC_BYTES is None:
        if os.path.isfile(_SRGB_ICC_PATH):
            with open(_SRGB_ICC_PATH, "rb") as f:
                _SRGB_ICC_BYTES = f.read()
        else:
            ImageCms = _import_cms()
            _SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            _SRGB_ICC_BYTES = _SRGB_PROFILE.tobytes()
    return _SRGB_ICC_BYTES


def _import_cms():
    try:
        from PIL import ImageCms  # noqa: PLC0415

        # 某些 Pillow 构建（如 conda 默认包）没有编译 _imagingcms，
        # 只有真正调用时才暴露 ImportError，因此这里做一次功能探测。
        ImageCms.createProfile("sRGB")
    except Exception as exc:  # pragma: no cover - 取决于 Pillow 编译选项
        raise PreprocessError(
            ErrorCode.ICC_ENGINE_UNAVAILABLE,
            Stage.COLOR,
            "当前 Pillow 未编译 littlecms，无法执行 ICC 色彩管理；"
            "请安装带 lcms 的 Pillow（PyPI 官方 wheel 自带）",
            detail=repr(exc),
        ) from exc
    return ImageCms


@dataclass
class StandardizedImage:
    """方向已落实、已转 sRGB 的 8-bit RGB 像素。"""

    pixels: np.ndarray
    format: str
    original_mode: str
    decoded_size: tuple[int, int]
    oriented_size: tuple[int, int]
    orientation: int
    orientation_applied: bool
    was_grayscale: bool
    icc_present: bool
    icc_description: str | None
    assumed_srgb: bool
    color_path: str
    exif_summary: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def load_standardized(path: str, cfg: dict[str, Any]) -> StandardizedImage:
    """完整解码并把像素标准化为 sRGB 8-bit RGB。"""
    warnings: list[str] = []

    try:
        with Image.open(path) as probe:
            fmt = (probe.format or "").upper()
            frames = int(getattr(probe, "n_frames", 1) or 1)
            probe.verify()
    except PreprocessError:
        raise
    except Exception as exc:  # Pillow 抛出的异常类型很杂
        raise PreprocessError(
            ErrorCode.DECODE_FAILED, Stage.DECODE, f"解码或完整性校验失败：{exc}", detail=repr(exc)
        ) from exc

    if fmt not in SUPPORTED_FORMATS:
        raise PreprocessError(
            ErrorCode.UNSUPPORTED_FORMAT,
            Stage.DECODE,
            f"内容格式不受支持：{fmt or '未知'}（仅支持 JPEG/PNG/TIFF/WebP）",
        )
    if frames > 1:
        raise PreprocessError(
            ErrorCode.MULTI_FRAME_UNSUPPORTED,
            Stage.DECODE,
            f"多帧/动图不受支持：{frames} 帧",
        )

    try:
        with Image.open(path) as im:
            im.load()
            original_mode = im.mode
            decoded_size = im.size
            icc_bytes = im.info.get("icc_profile")
            has_alpha = _has_alpha(im)
            exif, exif_warning = _read_exif(im)
            orientation = _read_orientation(exif)
            if exif_warning:
                warnings.append(exif_warning)

            if has_alpha:
                raise SkipImage(SkipReason.UNSUPPORTED_ALPHA, "含 Alpha 通道，首版不处理（不自行合成背景）")

            if orientation != 1:
                try:
                    transposed = ImageOps.exif_transpose(im)
                except Exception as exc:
                    raise PreprocessError(
                        ErrorCode.ORIENTATION_FAILED,
                        Stage.ORIENT,
                        f"应用 EXIF Orientation={orientation} 失败：{exc}",
                    ) from exc
                if transposed is None:  # 理论上不会发生
                    raise PreprocessError(ErrorCode.ORIENTATION_FAILED, Stage.ORIENT, "EXIF 方向校正返回空图像")
                working = transposed
                orientation_applied = True
            else:
                working = im
                orientation_applied = False

            oriented_size = working.size

            if max(oriented_size) < int(cfg["min_long_edge"]):
                raise SkipImage(
                    SkipReason.TOO_SMALL,
                    f"长边 {max(oriented_size)} px 低于最小可用长边 {cfg['min_long_edge']} px",
                )

            was_grayscale, bit_warnings = _is_grayscale(original_mode)
            warnings.extend(bit_warnings)

            rgb, color_path, icc_present, icc_desc, assumed_srgb, color_warnings, was_gray_converted = (
                _to_srgb(working, icc_bytes, original_mode)
            )
            warnings.extend(color_warnings)
            was_grayscale = was_grayscale or was_gray_converted

            pixels = np.asarray(rgb, dtype=np.uint8)
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise PreprocessError(
                    ErrorCode.UNSUPPORTED_COLOR_MODE,
                    Stage.COLOR,
                    f"标准化结果不是三通道 RGB：shape={pixels.shape}",
                )
    except SkipImage:
        raise
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(
            ErrorCode.DECODE_FAILED, Stage.DECODE, f"图像解码失败：{exc}", detail=repr(exc)
        ) from exc

    summary = None
    if cfg.get("include_exif_summary", True):
        summary = _build_exif_summary(exif)

    return StandardizedImage(
        pixels=pixels,
        format=fmt,
        original_mode=original_mode,
        decoded_size=(int(decoded_size[0]), int(decoded_size[1])),
        oriented_size=(int(oriented_size[0]), int(oriented_size[1])),
        orientation=orientation,
        orientation_applied=orientation_applied,
        was_grayscale=bool(was_grayscale),
        icc_present=bool(icc_present),
        icc_description=icc_desc,
        assumed_srgb=bool(assumed_srgb),
        color_path=color_path,
        exif_summary=summary,
        warnings=warnings,
    )


def _is_identity_srgb(icc_bytes: bytes | None, mode: str) -> bool:
    """内嵌配置是否就是本模块写入预览所用的标准 sRGB。

    仅当 ICC 字节与随包 sRGB 配置文件完全一致、且 ICC 头声明 RGB 空间、
    图像已是 RGB 时成立；任何不一致都必须走真正的 CMS 变换，绝不猜测。
    """
    if not icc_bytes or mode != "RGB":
        return False
    if len(icc_bytes) < 20 or icc_bytes[16:20] != b"RGB ":
        return False
    try:
        return icc_bytes == srgb_icc_bytes()
    except Exception:
        return False


def _has_alpha(im: Image.Image) -> bool:
    if im.mode in _ALPHA_MODES:
        return True
    if im.mode == "P" and "transparency" in im.info:
        return True
    return bool(im.info.get("transparency") is not None and im.mode in ("P", "L", "RGB"))


def _is_grayscale(mode: str) -> tuple[bool, list[str]]:
    if mode in _GRAY_MODES:
        return True, []
    return False, []


def _read_orientation(exif) -> int:
    try:
        raw = exif.get(274, 1) if exif is not None else 1
        value = int(raw)
    except Exception:
        return 1
    return value if 1 <= value <= 8 else 1


def _read_exif(im: Image.Image):
    try:
        exif = im.getexif()
        return exif, None
    except Exception:
        return None, WarningCode.EXIF_UNREADABLE


def _to_srgb(
    im: Image.Image, icc_bytes: bytes | None, original_mode: str
) -> tuple[Image.Image, str, bool, str | None, bool, list[str], bool]:
    """返回 (RGB 图像, 颜色路径, 是否有 ICC, ICC 描述, 是否假定 sRGB, 警告, 是否灰度转换)。"""
    warnings: list[str] = []
    was_gray = original_mode in _GRAY_MODES

    working, bit_warnings = _normalize_mode(im)
    warnings.extend(bit_warnings)

    icc_present = bool(icc_bytes)
    assumed_srgb = False
    icc_desc: str | None = None

    if icc_present:
        if _is_identity_srgb(icc_bytes, working.mode):
            # 内嵌配置与标准 sRGB 配置文件字节完全一致：源空间即目标空间，无需变换。
            # 这不是“静默更换色彩空间”，而是可证明的恒等转换。
            icc_desc = "sRGB IEC61966-2.1"
            color_path = "icc_srgb_identity"
        else:
            ImageCms = _import_cms()
            try:
                in_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            except Exception as exc:
                raise PreprocessError(
                    ErrorCode.ICC_CORRUPT, Stage.COLOR, f"内嵌 ICC 无法解析：{exc}", detail=repr(exc)
                ) from exc

            try:
                icc_desc = (ImageCms.getProfileDescription(in_profile) or "").strip() or None
            except Exception:
                warnings.append(WarningCode.ICC_DESCRIPTION_UNREADABLE)

            try:
                srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
                converted = ImageCms.profileToProfile(
                    working, in_profile, srgb_profile, renderingIntent=0, outputMode="RGB"
                )
            except Exception as exc:
                raise PreprocessError(
                    ErrorCode.ICC_TRANSFORM_FAILED,
                    Stage.COLOR,
                    f"ICC 转换到 sRGB 失败：{exc}",
                    detail=repr(exc),
                ) from exc
            color_path = "icc_to_srgb"
            working = converted
    else:
        if working.mode in ("CMYK", "LAB", "HSV", "YCbCr"):
            raise PreprocessError(
                ErrorCode.ICC_REQUIRED_FOR_MODE,
                Stage.COLOR,
                f"模式 {working.mode} 必须提供 ICC 才能转换到 sRGB",
            )
        assumed_srgb = True
        warnings.append(WarningCode.MISSING_ICC_ASSUMED_SRGB)
        color_path = "assumed_srgb"

    if working.mode != "RGB":
        try:
            working = working.convert("RGB")
        except Exception as exc:
            raise PreprocessError(
                ErrorCode.UNSUPPORTED_COLOR_MODE,
                Stage.COLOR,
                f"无法转换为 RGB：{original_mode} -> {working.mode}：{exc}",
            ) from exc

    return working, color_path, icc_present, icc_desc, assumed_srgb, warnings, was_gray


def _normalize_mode(im: Image.Image) -> tuple[Image.Image, list[str]]:
    """把非常规位深/调色板模式归一到 8-bit，供 ICC 变换使用。"""
    warnings: list[str] = []
    mode = im.mode

    if mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(im, dtype=np.float64)
        if arr.size and float(arr.max()) > 255.0:
            peak = 65535.0 if float(arr.max()) <= 65535.0 else float(arr.max())
            warnings.append(WarningCode.HIGH_BIT_DEPTH_SCALED)
            scaled = np.clip(np.floor(arr / peak * 255.0 + 0.5), 0, 255).astype(np.uint8)
        else:
            scaled = np.clip(np.floor(arr + 0.5), 0, 255).astype(np.uint8)
        return Image.fromarray(scaled, mode="L"), warnings

    if mode == "P":
        return im.convert("RGB"), warnings

    if mode == "1":
        return im.convert("L"), warnings

    return im, warnings


def _build_exif_summary(exif) -> dict[str, Any] | None:
    """白名单读取 EXIF 摘要，绝不触碰 GPS / MakerNote。"""
    if exif is None:
        return None

    try:
        sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
    except Exception:
        sub = {}

    def pick(tag: int):
        value = sub.get(tag)
        if value is None:
            value = exif.get(tag)
        return value

    def as_text(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                return None
        text = str(value).replace("\x00", "").strip()
        return text or None

    def as_float(value) -> float | None:
        if value is None:
            return None
        try:
            return round_float(float(value))
        except Exception:
            return None

    def as_int(value) -> int | None:
        if value is None:
            return None
        try:
            if isinstance(value, (tuple, list)):
                value = value[0]
            return int(value)
        except Exception:
            return None

    make = as_text(pick(271))
    model = as_text(pick(272))
    lens = as_text(pick(42036)) or as_text(pick(42037))
    iso = as_int(pick(34855))
    exposure = as_float(pick(33434))
    f_number = as_float(pick(33437))
    focal = as_float(pick(37386))
    taken = as_text(pick(36867)) or as_text(pick(306))

    summary: dict[str, Any] = {
        "camera_make": make,
        "camera_model": model,
        "lens_model": lens,
        "iso": iso,
        "exposure_time_seconds": exposure,
        "f_number": f_number,
        "focal_length_mm": focal,
        "datetime_original": taken,
    }
    if exposure is not None and exposure > 0:
        if exposure < 1:
            summary["exposure_time"] = f"1/{int(round(1.0 / exposure))}"
        else:
            summary["exposure_time"] = f"{exposure:g}s"

    if not any(v is not None for v in summary.values()):
        return None
    return summary
