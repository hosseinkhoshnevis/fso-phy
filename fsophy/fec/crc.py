"""CRCs of the OCT modem frame: CRC-16 (CCITT X.25 polynomial) over the
header fields, CRC-32 (IEEE 802.3 polynomial) over the payload. The
standard initialises both registers to zero at the start of each
calculation, so that is what we do; there is no final inversion. Bits
are processed MSB-first, batch-vectorised over frames.
"""

import numpy as np

POLY16 = 0x1021
POLY32 = 0x04C11DB7


def _crc(bits, poly, width):
    bits = np.asarray(bits, dtype=np.uint32)
    if bits.ndim == 1:
        bits = bits[None]
    reg = np.zeros(bits.shape[0], dtype=np.uint64)
    top = np.uint64(1 << (width - 1))
    mask = np.uint64((1 << width) - 1)
    p = np.uint64(poly)
    for k in range(bits.shape[1]):
        fb = ((reg & top) != 0) ^ (bits[:, k] != 0)
        reg = (reg << np.uint64(1)) & mask
        reg = np.where(fb, reg ^ p, reg)
    return reg


def crc16(bits):
    """(B, n) or (n,) bits -> (B, 16) CRC bits, MSB first."""
    reg = _crc(bits, POLY16, 16)
    return ((reg[:, None] >> np.arange(15, -1, -1).astype(np.uint64)) & np.uint64(1)).astype(np.uint8)


def crc32(bits):
    """(B, n) or (n,) bits -> (B, 32) CRC bits, MSB first."""
    reg = _crc(bits, POLY32, 32)
    return ((reg[:, None] >> np.arange(31, -1, -1).astype(np.uint64)) & np.uint64(1)).astype(np.uint8)
