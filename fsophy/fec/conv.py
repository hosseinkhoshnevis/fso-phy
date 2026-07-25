"""The header FEC: non-systematic rate-1/6, constraint-length-7
convolutional code, generators (octal) 175, 171, 151, 133, 127, 117,
zero-tailed. 160 bits in (128 fields + CRC-16 + 16 tail zeros), 960
bits out. Six-fold redundancy is the standard's way of making the
header decodable well below the payload threshold, so a receiver knows
the frame's PL_RATE before it can decode the frame.

The Viterbi decoder is soft-input and batch-vectorised over frames:
the trellis loop runs over the 160 steps in Python, but every step
updates all 64 states of all frames in one numpy expression.
"""

import numpy as np

GENS = (0o175, 0o171, 0o151, 0o133, 0o127, 0o117)
K = 7
N_STATES = 64


def _out_table():
    """out[state, bit] -> 6 coded bits; state is the last 6 inputs,
    newest in the MSB."""
    out = np.zeros((N_STATES, 2, 6), dtype=np.uint8)
    for s in range(N_STATES):
        for b in range(2):
            reg = (b << 6) | s
            for gi, g in enumerate(GENS):
                out[s, b, gi] = bin(reg & g).count("1") & 1
    return out


_OUT = _out_table()
# next state: shift the newest bit into the MSB of the 6-bit state
_NEXT = np.array([[(s >> 1) | (b << 5) for b in range(2)]
                  for s in range(N_STATES)])


def encode(bits):
    """(B, 160) -> (B, 960)."""
    bits = np.asarray(bits, dtype=np.uint8)
    B, n = bits.shape
    out = np.zeros((B, n, 6), dtype=np.uint8)
    state = np.zeros(B, dtype=np.int64)
    for t in range(n):
        out[:, t] = _OUT[state, bits[:, t]]
        state = _NEXT[state, bits[:, t]]
    return out.reshape(B, 6 * n)


def viterbi(llrs):
    """Soft decode. llrs (B, 960), positive = bit 0 -> (B, 160)."""
    llrs = np.asarray(llrs, dtype=np.float32).reshape(-1, 160, 6)
    B = llrs.shape[0]
    signs = (1.0 - 2.0 * _OUT.astype(np.float32))          # (64, 2, 6)
    pm = np.full((B, N_STATES), -1e9, dtype=np.float32)
    pm[:, 0] = 0.0
    back = np.zeros((B, 160, N_STATES), dtype=np.uint8)
    prev0 = np.array([[s for s in range(N_STATES) for b in range(2)
                       if _NEXT[s, b] == ns] for ns in range(N_STATES)])
    prev_bit = np.array([[b for s in range(N_STATES) for b in range(2)
                          if _NEXT[s, b] == ns] for ns in range(N_STATES)])
    for t in range(160):
        bm = np.einsum("bk,spk->bsp", llrs[:, t], signs)   # (B, 64, 2)
        cand = pm[:, prev0] + bm[:, prev0, prev_bit]       # (B, 64, 2)
        choice = cand.argmax(axis=2)
        pm = np.take_along_axis(cand, choice[..., None], axis=2)[..., 0]
        back[:, t] = choice.astype(np.uint8)
    bits = np.zeros((B, 160), dtype=np.uint8)
    state = np.zeros(B, dtype=np.int64)                    # zero tail ends at 0
    for t in range(159, -1, -1):
        c = back[np.arange(B), t, state]
        bits[:, t] = prev_bit[state, c]
        state = prev0[state, c]
    return bits
