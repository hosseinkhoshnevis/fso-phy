"""Frame structure against the standard's numbers."""

import numpy as np

from fsophy import framing


def test_preamble_is_the_standard_word_msb_first():
    assert framing.PREAMBLE.size == 64
    # 0x53 = 0101 0011 leads the wire
    assert list(framing.PREAMBLE[:8]) == [0, 1, 0, 1, 0, 0, 1, 1]
    val = 0
    for b in framing.PREAMBLE:
        val = (val << 1) | int(b)
    assert val == framing.PREAMBLE_HEX


def test_scrambler_period_and_self_inverse():
    seq = framing._SCR
    assert seq.size == 32767 and seq.sum() == 16384   # maximal length
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, (2, 40000), dtype=np.uint8)
    assert (framing.scramble(framing.scramble(x)) == x).all()


def test_frame_lengths_per_pl_rate():
    assert framing.frame_bits(0) == 64 + 960 + 8448
    assert framing.frame_bits(1) == 64 + 960 + 7680 + 2304
    assert framing.frame_bits(4) == 64 + 960 + 7680 + 9216


def test_build_and_parse_roundtrip():
    rng = np.random.default_rng(1)
    payload = rng.integers(0, 2, (3, framing.PAYLOAD_BITS), dtype=np.uint8)
    line, body = framing.build_frames(payload, 4)
    assert line.shape[1] == framing.frame_bits(4)
    assert (line[:, :64] == framing.PREAMBLE).all()
    # perfect channel: LLR = +-1 from the line bits after the preamble
    llr = 1.0 - 2.0 * line[:, 64:].astype(np.float32)
    hdr, got_body = framing.split_frame(llr, 4)
    assert ((got_body < 0).astype(np.uint8) == body).all()
    _, ok, pl, idx = framing.parse_header(hdr)
    assert ok.all() and (pl == 4).all() and (idx == np.arange(3)).all()
