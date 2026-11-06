"""合成测试语料（不依赖任何旧训练集）。

覆盖需求文档 §12 的最小测试集：尺寸/方向/ICC/灰度/高位深/Alpha/损坏/伪扩展名/
过小/完全重复/改名重复/近重复/纯色/低饱和/高饱和/中文路径/极端长宽比/多帧。
"""

from __future__ import annotations

import os
import shutil

import numpy as np
from PIL import Image, features as pil_features

HAS_LCMS = bool(pil_features.check("littlecms2"))
_SRGB_ICC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "srgb.icc")


def srgb_icc_bytes() -> bytes | None:
    if not os.path.isfile(_SRGB_ICC_PATH):
        return None
    with open(_SRGB_ICC_PATH, "rb") as f:
        return f.read()


def gradient(width: int, height: int) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.float64)
    y = np.linspace(0, 255, height, dtype=np.float64)[:, None]
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.clip(x, 0, 255).astype(np.uint8)
    arr[..., 1] = np.clip(y, 0, 255).astype(np.uint8)
    arr[..., 2] = np.clip((x[None, :] + y) / 2, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def build_full_corpus(root: str) -> dict:
    """生成完整测试语料，返回路径字典。"""
    inbox = os.path.join(root, "输入 目录")
    sub = os.path.join(inbox, "子目录 中文")
    os.makedirs(sub, exist_ok=True)
    icc = srgb_icc_bytes()

    paths: dict[str, str] = {}

    # 1) 正常横图 / 竖图 / 正方形
    p = os.path.join(inbox, "风景 横图.JPG")
    gradient(3000, 2000).save(p, quality=92)
    paths["landscape"] = p

    p = os.path.join(inbox, "竖图.png")
    gradient(2000, 3000).save(p)
    paths["portrait"] = p

    p = os.path.join(inbox, "square.jpg")
    gradient(1200, 1200).save(p, quality=92)
    paths["square"] = p

    # 2) EXIF Orientation=6（存储 1200x800，显示应为 800x1200）
    p = os.path.join(inbox, "orient6.jpg")
    exif = Image.Exif()
    exif[274] = 6
    gradient(1200, 800).save(p, quality=92, exif=exif.tobytes())
    paths["orient6"] = p

    # 3) ICC / 无 ICC / 灰度 / 16-bit
    if icc:
        p = os.path.join(inbox, "icc_srgb.jpg")
        gradient(900, 700).save(p, quality=92, icc_profile=icc)
        paths["icc_srgb"] = p

        p = os.path.join(inbox, "icc_lab_mismatch.jpg")
        try:
            from PIL import ImageCms

            lab_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
            gradient(800, 600).save(p, quality=92, icc_profile=lab_profile)
            paths["icc_broken"] = p
        except Exception:
            pass

    p = os.path.join(inbox, "gray.jpg")
    gray = np.tile(np.linspace(0, 255, 900, dtype=np.uint8), (700, 1))
    Image.fromarray(gray, "L").save(p, quality=92)
    paths["gray"] = p

    p = os.path.join(inbox, "deep16.png")
    arr16 = np.tile(np.linspace(0, 65535, 900, dtype=np.uint16), (700, 1))
    Image.fromarray(arr16).save(p)
    paths["deep16"] = p

    # 4) Alpha / 损坏 / 截断 / 伪扩展名 / 过小 / 多帧
    p = os.path.join(inbox, "alpha.png")
    rgba = np.zeros((600, 800, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 128
    Image.fromarray(rgba, "RGBA").save(p)
    paths["alpha"] = p

    p = os.path.join(inbox, "corrupt.jpg")
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"garbage" * 200)
    paths["corrupt"] = p

    p = os.path.join(inbox, "truncated.jpg")
    tmp = os.path.join(root, "_tmp_full.jpg")
    gradient(1400, 900).save(tmp, quality=92)
    with open(tmp, "rb") as f:
        raw = f.read()
    with open(p, "wb") as f:
        f.write(raw[: len(raw) // 2])
    os.unlink(tmp)
    paths["truncated"] = p

    p = os.path.join(inbox, "伪装成jpg的png.jpg")
    gradient(850, 620).save(p, format="PNG")
    paths["disguised"] = p

    p = os.path.join(inbox, "notimage.jpg")
    with open(p, "w", encoding="utf-8") as f:
        f.write("这其实是一个文本文件，不是图片。")
    paths["notimage"] = p

    p = os.path.join(inbox, "过小.jpg")
    gradient(300, 200).save(p, quality=92)
    paths["too_small"] = p

    p = os.path.join(inbox, "多帧.tif")
    a = gradient(800, 600)
    b = gradient(800, 600).transpose(Image.Transpose.ROTATE_180)
    a.save(p, save_all=True, append_images=[b])
    paths["multiframe"] = p

    p = os.path.join(inbox, "fake_gif.jpg")
    gradient(700, 500).save(p, format="GIF")
    paths["fake_gif"] = p

    # 5) 完全重复 / 改名重复 / 近重复
    base = os.path.join(inbox, "dup_a.jpg")
    gradient(900, 600).save(base, quality=92)
    paths["dup_a"] = base

    p = os.path.join(inbox, "dup_copy.jpg")
    shutil.copyfile(base, p)
    paths["dup_copy"] = p

    p = os.path.join(inbox, "改名 重复.JPG")
    shutil.copyfile(base, p)
    paths["dup_rename"] = p

    near_a = os.path.join(inbox, "near_a.png")
    gradient(800, 600).save(near_a)
    paths["near_a"] = near_a
    near_b = os.path.join(inbox, "near_b.png")
    img = np.array(Image.open(near_a).convert("RGB"))
    img[0:10, 0:10] = 255 - img[0:10, 0:10]
    Image.fromarray(img).save(near_b)
    paths["near_b"] = near_b

    # 6) 纯黑/纯白/纯红/灰阶/低饱和/高饱和
    p = os.path.join(inbox, "纯黑.png")
    Image.new("RGB", (800, 600), (0, 0, 0)).save(p)
    paths["black"] = p

    p = os.path.join(inbox, "纯白.png")
    Image.new("RGB", (800, 600), (255, 255, 255)).save(p)
    paths["white"] = p

    p = os.path.join(inbox, "纯红.png")
    Image.new("RGB", (800, 600), (255, 0, 0)).save(p)
    paths["red"] = p

    p = os.path.join(inbox, "灰阶渐变.png")
    ramp = np.tile(np.linspace(0, 255, 900, dtype=np.uint8), (600, 1))
    Image.fromarray(np.dstack([ramp, ramp, ramp])).save(p)
    paths["gradient_gray"] = p

    p = os.path.join(inbox, "低饱和.png")
    low = np.zeros((600, 800, 3), dtype=np.uint8)
    low[:] = (128, 130, 129)
    low[:, 400:] = (120, 122, 121)
    Image.fromarray(low).save(p)
    paths["low_sat"] = p

    p = os.path.join(inbox, "高饱和.png")
    high = np.zeros((600, 800, 3), dtype=np.uint8)
    high[:, :200] = (255, 0, 0)
    high[:, 200:400] = (0, 255, 0)
    high[:, 400:600] = (0, 0, 255)
    high[:, 600:] = (255, 255, 0)
    Image.fromarray(high).save(p)
    paths["high_sat"] = p

    p = os.path.join(inbox, "noise.png")
    rng = np.random.RandomState(7)
    Image.fromarray(rng.randint(0, 256, (600, 800, 3), dtype=np.uint8)).save(p)
    paths["noise"] = p

    # 7) 极端长宽比
    p = os.path.join(inbox, "超长图.png")
    line = np.zeros((1, 5000, 3), dtype=np.uint8)
    line[0, :, 0] = np.linspace(0, 255, 5000, dtype=np.uint8)
    line[0, :, 1] = 40
    Image.fromarray(line).save(p)
    paths["extreme"] = p

    # 8) 中文子目录 + 空格 + 大写扩展名 + EXIF 摘要
    p = os.path.join(sub, "带 空格 名字.PNG")
    gradient(900, 700).save(p)
    paths["cn_subdir"] = p

    p = os.path.join(inbox, "exif_meta.jpg")
    exif = Image.Exif()
    exif[271] = "TestMake"
    exif[272] = "TestModel"
    exif[34855] = 200
    exif[33434] = 1 / 250
    exif[33437] = 2.8
    exif[37386] = 35.0
    gradient(900, 600).save(p, quality=92, exif=exif.tobytes())
    paths["exif_meta"] = p

    return {"inbox": inbox, "paths": paths}


def build_small_corpus(root: str) -> dict:
    inbox = os.path.join(root, "小 输入")
    os.makedirs(inbox, exist_ok=True)
    a = os.path.join(inbox, "a.jpg")
    gradient(700, 500).save(a, quality=92)
    b = os.path.join(inbox, "b.png")
    gradient(640, 480).save(b)
    return {"inbox": inbox, "paths": {"a": a, "b": b}}
