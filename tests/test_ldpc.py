"""The payload code against the standard's numbers."""

import numpy as np
import pytest

from fsophy.fec import ldpc


def test_dimensions_match_the_standard():
    assert ldpc.Z == 384
    assert ldpc.K_BITS == 8448
    assert ldpc.N_ROWS == 46 and ldpc.N_COLS == 68
    assert ldpc.PL_PARITY_BITS == {1: 2304, 2: 3456, 3: 4992, 4: 9216}


def test_table_3_11_code_rates():
    for pl, parity in ldpc.PL_PARITY_BITS.items():
        rate = 8448 / (8448 - 768 + parity)
        expected = {1: 0.8462, 2: 0.7586, 3: 0.6667, 4: 0.5000}[pl]
        assert abs(rate - expected) < 5e-4


def test_encoder_makes_valid_codewords():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, (3, ldpc.K_BITS), dtype=np.uint8)
    assert ldpc.check(ldpc.encode(u)) == 0


def test_rate_match_punctures_first_two_blocks():
    rng = np.random.default_rng(1)
    u = rng.integers(0, 2, (1, ldpc.K_BITS), dtype=np.uint8)
    cw = ldpc.encode(u)
    tx = ldpc.rate_match(cw, 4)
    assert tx.shape[1] == 7680 + 9216
    assert (tx[0, :7680] == u[0, 768:]).all()


@pytest.mark.parametrize("pl", [1, 2, 3, 4])
def test_clean_decode_every_rate(pl):
    rng = np.random.default_rng(pl)
    u = rng.integers(0, 2, (2, ldpc.K_BITS), dtype=np.uint8)
    tx = ldpc.rate_match(ldpc.encode(u), pl)
    llr = 8.0 * (1.0 - 2.0 * tx.astype(np.float32))
    info, ok = ldpc.Decoder(pl).decode(llr)
    assert ok.all() and (info == u).all()


def test_soft_decode_inside_the_waterfall():
    rng = np.random.default_rng(2)
    u = rng.integers(0, 2, (2, ldpc.K_BITS), dtype=np.uint8)
    tx = ldpc.rate_match(ldpc.encode(u), 4)
    sigma = 0.8
    x = (1 - 2.0 * tx.astype(np.float32)) + sigma * rng.standard_normal(tx.shape).astype(np.float32)
    raw = float(((x < 0) != (tx > 0)).mean())
    info, ok = ldpc.Decoder(4).decode(2 * x / sigma ** 2)
    assert raw > 0.08                     # genuinely noisy input
    assert ok.all() and (info == u).all()


def test_spa_option_decodes_inside_the_waterfall():
    rng = np.random.default_rng(5)
    u = rng.integers(0, 2, (2, ldpc.K_BITS), dtype=np.uint8)
    tx = ldpc.rate_match(ldpc.encode(u), 4)
    sigma = 0.8
    x = (1 - 2.0 * tx.astype(np.float32)) + sigma * rng.standard_normal(tx.shape).astype(np.float32)
    info, ok = ldpc.Decoder(4, alg="spa").decode(2 * x / sigma ** 2)
    assert ok.all() and (info == u).all()


def test_hard_input_decoder_corrects_flips():
    rng = np.random.default_rng(3)
    u = rng.integers(0, 2, (2, ldpc.K_BITS), dtype=np.uint8)
    tx = ldpc.rate_match(ldpc.encode(u), 4)
    flips = (rng.random(tx.shape) < 0.05).astype(np.uint8)
    info, ok = ldpc.Decoder(4).decode_hard(tx ^ flips)
    assert ok.all() and (info == u).all()
