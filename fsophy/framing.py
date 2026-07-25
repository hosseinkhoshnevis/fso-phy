"""The OCT modem frame: preamble, coded header, rate-matched payload,
and the frame scrambler.

On the wire a frame is

    64-bit preamble | 960-bit coded header | 7680 + parity payload bits

with everything after the preamble XORed with the standard's scrambler,
a maximal-length sequence from x^15 + x^14 + 1 restarted every frame
from the fixed seed. Frames are back to back with no gaps. The header
carries 128 field bits (we model frame count, PL_RATE and encoding;
the standard's full field map has more), then CRC-16 and a 16-bit zero
tail into the rate-1/6 convolutional code.
"""

import numpy as np

from .fec import conv, crc, ldpc

PREAMBLE_HEX = 0x53225B1D0D73DF03
PREAMBLE = np.array([(PREAMBLE_HEX >> k) & 1 for k in range(63, -1, -1)],
                    dtype=np.uint8)          # MSB transmitted first
PAYLOAD_BITS = 8416                          # info bits per frame
HEADER_CODED = 960
SCRAMBLER_SEED = [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0]


def _scrambler_period():
    reg = list(SCRAMBLER_SEED)
    out = np.empty(32767, dtype=np.uint8)
    for k in range(32767):
        out[k] = reg[14]
        fb = reg[14] ^ reg[13]
        reg = [fb] + reg[:14]
    return out


_SCR = _scrambler_period()


def scramble(bits):
    """XOR with the scrambler sequence, restarted per frame. Works on
    (B, n) frames; self-inverse."""
    n = bits.shape[-1]
    reps = int(np.ceil(n / 32767))
    seq = np.tile(_SCR, reps)[:n]
    return bits ^ seq


def frame_bits(pl_rate):
    """Total bits on the wire for one frame at a PL_RATE."""
    if pl_rate == 0:
        return 64 + HEADER_CODED + ldpc.K_BITS
    return 64 + HEADER_CODED + (ldpc.K_BITS - 2 * ldpc.Z) + ldpc.PL_PARITY_BITS[pl_rate]


def build_header(frame_idx, pl_rate, encoding):
    """128 field bits -> CRC-16 -> zero tail -> rate-1/6 CC -> 960 bits."""
    B = len(frame_idx)
    fields = np.zeros((B, 128), dtype=np.uint8)
    fc = np.asarray(frame_idx, dtype=np.int64) & 0xFFFFFFFF
    fields[:, :32] = (fc[:, None] >> np.arange(31, -1, -1)) & 1
    fields[:, 32:35] = (np.full(B, pl_rate)[:, None] >> np.arange(2, -1, -1)) & 1
    fields[:, 35] = 1 if encoding == "manchester" else 0
    with_crc = np.concatenate([fields, crc.crc16(fields)], axis=1)
    tailed = np.concatenate([with_crc, np.zeros((B, 16), np.uint8)], axis=1)
    return conv.encode(tailed), fields


def parse_header(header_llrs):
    """Soft 960 -> (fields, crc_ok, pl_rate, frame_idx)."""
    dec = conv.viterbi(header_llrs)
    fields, rx_crc = dec[:, :128], dec[:, 128:144]
    ok = (crc.crc16(fields) == rx_crc).all(axis=1)
    pl = fields[:, 32:35].dot(1 << np.arange(2, -1, -1))
    idx = fields[:, :32].astype(np.int64).dot(1 << np.arange(31, -1, -1, dtype=np.int64))
    return fields, ok, pl, idx


def build_frames(payload, pl_rate, encoding="nrz"):
    """payload (B, 8416) -> line bits (B, frame_bits) plus the coded
    reference the receiver's pre-FEC BER is counted against."""
    B = payload.shape[0]
    body_in = np.concatenate([payload, crc.crc32(payload)], axis=1)  # 8448
    if pl_rate == 0:
        body = body_in
    else:
        body = ldpc.rate_match(ldpc.encode(body_in), pl_rate)
    header, _ = build_header(np.arange(B), pl_rate, encoding)
    unscrambled = np.concatenate([header, body], axis=1)
    line = np.concatenate([np.tile(PREAMBLE, (B, 1)), scramble(unscrambled)], axis=1)
    return line, body


def split_frame(llrs, pl_rate):
    """Descramble a frame's post-preamble LLRs and split header/body.
    Scrambling in the LLR domain is a sign flip where the sequence is 1."""
    n = llrs.shape[-1]
    reps = int(np.ceil(n / 32767))
    seq = np.tile(_SCR, reps)[:n]
    d = llrs * (1.0 - 2.0 * seq)
    return d[:, :HEADER_CODED], d[:, HEADER_CODED:]
