"""Header code and the two CRCs."""

import numpy as np

from fsophy.fec import conv, crc


def test_conv_expands_six_to_one():
    bits = np.zeros((1, 160), dtype=np.uint8)
    assert conv.encode(bits).shape == (1, 960)
    # all-zero input through a feedforward code is all-zero output
    assert conv.encode(bits).sum() == 0


def test_viterbi_survives_heavy_errors():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, (4, 144), dtype=np.uint8)
    coded = conv.encode(np.concatenate([b, np.zeros((4, 16), np.uint8)], axis=1))
    llr = 1.0 - 2.0 * coded.astype(np.float32)
    llr[:, ::9] *= -1.0                    # 11% of the bits inverted
    dec = conv.viterbi(llr)
    assert (dec[:, :144] == b).all()


def test_crcs_are_linear_and_detect_flips():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, (1, 128), dtype=np.uint8)
    b = rng.integers(0, 2, (1, 128), dtype=np.uint8)
    # zero-initialised CRC is a linear map
    assert (crc.crc16(a ^ b) == crc.crc16(a) ^ crc.crc16(b)).all()
    p = rng.integers(0, 2, (1, 8416), dtype=np.uint8)
    ref = crc.crc32(p)
    for k in (0, 4210, 8415):
        q = p.copy()
        q[0, k] ^= 1
        assert (crc.crc32(q) != ref).any()
