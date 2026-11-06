"""感知哈希（64 位 DCT pHash）。

算法固定为：
1. 相对亮度 → 8-bit 灰度，双线性缩放到 32×32；
2. 二维 DCT-II；
3. 取左上 8×8 低频块，用除 DC 外 63 个系数的中位数做阈值；
4. 逐位比较得到 64 bit，输出 16 位小写十六进制。

近重复只用于提示人工复核，不能据此判定内容相同或自动删除。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PHASH_ALGORITHM = "dct_phash_64_v1"
_DCT_SIZE = 32
_LOW = 8


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)[:, None]
    x = np.arange(n, dtype=np.float64)[None, :]
    matrix = np.cos(np.pi * (2.0 * x + 1.0) * k / (2.0 * n)) * np.sqrt(2.0 / n)
    matrix[0, :] = np.sqrt(1.0 / n)
    return matrix


_DCT = _dct_matrix(_DCT_SIZE)


def phash64(luma_u8: np.ndarray) -> str:
    """luma_u8: HxW uint8 相对亮度。"""
    small = Image.fromarray(np.ascontiguousarray(luma_u8, dtype=np.uint8)).resize(
        (_DCT_SIZE, _DCT_SIZE), Image.Resampling.BILINEAR
    )
    block = np.asarray(small, dtype=np.float64)
    coeffs = _DCT @ block @ _DCT.T
    low = coeffs[:_LOW, :_LOW].ravel()
    threshold = float(np.median(low[1:]))
    bits = low > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """两个 16 位十六进制 pHash 的 Hamming 距离（0..64）。"""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()
