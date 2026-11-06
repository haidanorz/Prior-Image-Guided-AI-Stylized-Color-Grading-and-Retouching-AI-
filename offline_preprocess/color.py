"""颜色数学：sRGB 线性化、CIE Lab(D65)、HSV 饱和度、亮度。

所有函数只做纯计算，不读写文件；输入输出均为 numpy 数组。
内部统一用 float64，避免多次量化；对外输出时再统一保留 6 位小数。
"""

from __future__ import annotations

import numpy as np

# sRGB (IEC 61966-2-1) 线性化查找表，输入 8-bit 值
_LINEAR_LUT = np.where(
    np.arange(256, dtype=np.float64) / 255.0 <= 0.04045,
    (np.arange(256, dtype=np.float64) / 255.0) / 12.92,
    ((np.arange(256, dtype=np.float64) / 255.0 + 0.055) / 1.055) ** 2.4,
)

# sRGB -> XYZ (D65)，IEC 61966-2-1 标准矩阵
_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)

_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_EPS = (6.0 / 29.0) ** 3
_KAPPA = 3.0 * (6.0 / 29.0) ** 2
_OFFSET = 4.0 / 29.0


def linearize_u8(channel_u8: np.ndarray) -> np.ndarray:
    """8-bit sRGB 通道 -> 线性光 [0,1]。"""
    return _LINEAR_LUT[channel_u8]


def rgb_to_luminance(rgb_u8: np.ndarray) -> np.ndarray:
    """相对亮度 Y（CIE XYZ 的 Y，0..1），Rec.709 系数，作用于线性光。"""
    r = _LINEAR_LUT[rgb_u8[..., 0]]
    g = _LINEAR_LUT[rgb_u8[..., 1]]
    b = _LINEAR_LUT[rgb_u8[..., 2]]
    return 0.2126729 * r + 0.7151522 * g + 0.0721750 * b


def rgb_to_lab(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8-bit sRGB -> CIE Lab (D65)。返回 (L, a, b) 三个 HxW 数组。"""
    linear = np.empty(rgb_u8.shape, dtype=np.float64)
    linear[..., 0] = _LINEAR_LUT[rgb_u8[..., 0]]
    linear[..., 1] = _LINEAR_LUT[rgb_u8[..., 1]]
    linear[..., 2] = _LINEAR_LUT[rgb_u8[..., 2]]

    xyz = linear @ _M_RGB2XYZ.T
    t = xyz / _WHITE_D65
    f = np.where(t > _EPS, np.cbrt(t), t / _KAPPA + _OFFSET)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return lightness, a, b


def lab_to_srgb8(lab: np.ndarray) -> np.ndarray:
    """Lab(D65) -> 8-bit sRGB，用于主色板 HEX。lab 形状 (..., 3)。"""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    def inv_f(v: np.ndarray) -> np.ndarray:
        return np.where(v > 6.0 / 29.0, v**3, _KAPPA * (v - _OFFSET))

    xyz = np.stack([inv_f(fx), inv_f(fy), inv_f(fz)], axis=-1) * _WHITE_D65
    linear = xyz @ _M_XYZ2RGB.T
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return np.clip(np.floor(srgb * 255.0 + 0.5), 0, 255).astype(np.uint8)


def saturation_hsv(rgb_u8: np.ndarray) -> np.ndarray:
    """HSV 饱和度 S = (max-min)/max，max=0 时取 0。"""
    rgb = rgb_u8.astype(np.float64) / 255.0
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    denom = np.where(cmax > 0, cmax, 1.0)
    s = np.where(cmax > 0, (cmax - cmin) / denom, 0.0)
    return s


def percentile_values(values: np.ndarray, percents: tuple[float, ...]) -> list[float]:
    """线性插值分位数（numpy 默认 method='linear'），一次调用完成多分位点。"""
    flat = np.asarray(values, dtype=np.float64).ravel()
    return [float(v) for v in np.percentile(flat, percents, method="linear")]


def histogram_normalized(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> list[float]:
    """归一化直方图；越界值按端点归类，保证总和为 1。"""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return [0.0] * bins
    lo, hi = value_range
    scaled = np.floor((flat - lo) / (hi - lo) * bins).astype(np.int64)
    idx = np.clip(scaled, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(np.float64)
    return [float(c) for c in counts / flat.size]
