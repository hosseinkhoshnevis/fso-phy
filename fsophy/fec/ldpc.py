"""The payload FEC: 5G NR LDPC, base graph 1, Z = 384, as adopted by the
SDA OCT standard.

The standard reuses the terrestrial 5G data-channel code unchanged: info
block K = 8448 bits (22 blocks of Z = 384), and per PL_RATE it transmits
the first 6, 9, 13 or 24 parity blocks (2304 / 3456 / 4992 / 9216 bits,
rates 0.8462 / 0.7586 / 0.6667 / 0.5000). As in 5G, the first two
systematic blocks (768 bits) are never transmitted; the decoder recovers
them from the code, which is why the rates work out to those values.

Everything here is vectorised over a batch of codewords. The circulant
structure makes that natural: a lifted edge is one np.roll, and the Z
parallel check equations of a block row process elementwise once every
message is rolled into the check's index domain.
"""

import numpy as np

from .bg1 import SHIFTS, Z

N_ROWS = len(SHIFTS)          # 46
N_COLS = len(SHIFTS[0])       # 68
K_BLOCKS = 22
K_BITS = K_BLOCKS * Z         # 8448

# parity blocks transmitted for PL_RATE 1..4 (0 is uncoded)
PL_PARITY_BLOCKS = {1: 6, 2: 9, 3: 13, 4: 24}
PL_PARITY_BITS = {r: b * Z for r, b in PL_PARITY_BLOCKS.items()}

# flat edge list sorted by row: (row, col, shift)
EDGES = [(i, j, s) for i, row in enumerate(SHIFTS)
         for j, s in enumerate(row) if s >= 0]
_E_ROW = np.array([e[0] for e in EDGES])
_E_COL = np.array([e[1] for e in EDGES])
_E_SHIFT = np.array([e[2] for e in EDGES])
_ROW_PTR = np.searchsorted(_E_ROW, np.arange(N_ROWS + 1))


def _mul(shift, x):
    """P^shift applied to the last-but-one axis (..., Z, B)."""
    return np.roll(x, -shift, axis=-2)


def encode(info):
    """info (B, 8448) uint8 -> full codeword (B, 68*384) uint8.

    Core parities (blocks 22..25) come from the double-diagonal trick:
    summing the four core rows cancels every parity term but one shift of
    block 22, which solves it; the rest back-substitute. Extension
    parities are then direct sums of their rows.
    """
    info = np.asarray(info, dtype=np.uint8)
    B = info.shape[0]
    u = info.reshape(B, K_BLOCKS, Z).transpose(1, 2, 0)   # (22, Z, B)
    lam = np.zeros((4, Z, B), dtype=np.uint8)             # info part of core rows
    for i in range(4):
        for j in range(K_BLOCKS):
            s = SHIFTS[i][j]
            if s >= 0:
                lam[i] ^= _mul(s, u[j])
    # block 22: the odd shift survives the four-row sum
    s22 = [SHIFTS[i][22] for i in range(4) if SHIFTS[i][22] >= 0]
    surv = [s for s in set(s22) if s22.count(s) % 2 == 1]
    assert len(surv) == 1, "core structure changed?"
    q = np.zeros((4, Z, B), dtype=np.uint8)               # parities 22..25
    q[0] = np.roll(lam[0] ^ lam[1] ^ lam[2] ^ lam[3], surv[0], axis=0)
    known = [True, False, False, False]
    for _ in range(3):                                    # back-substitute
        for i in range(4):
            terms, unknown = lam[i].copy(), None
            for c in (22, 23, 24, 25):
                s = SHIFTS[i][c]
                if s < 0:
                    continue
                if known[c - 22]:
                    terms ^= _mul(s, q[c - 22])
                elif unknown is None:
                    unknown = (c - 22, s)
                else:
                    unknown = "two"
            if unknown not in (None, "two"):
                idx, s = unknown
                q[idx] = np.roll(terms, s, axis=0)        # P^s q = terms
                known[idx] = True
    assert all(known)
    cols = np.zeros((N_COLS, Z, B), dtype=np.uint8)
    cols[:K_BLOCKS] = u
    cols[22:26] = q
    for i in range(4, N_ROWS):                            # extension parities
        acc = np.zeros((Z, B), dtype=np.uint8)
        for j in range(26):
            s = SHIFTS[i][j]
            if s >= 0:
                acc ^= _mul(s, cols[j])
        cols[26 + (i - 4)] = acc                          # its own shift is 0
    return cols.transpose(2, 0, 1).reshape(B, N_COLS * Z)


def check(codeword):
    """Syndrome weight of full codewords (B, 68*384); 0 means valid."""
    c = np.asarray(codeword, dtype=np.uint8)
    cols = c.reshape(c.shape[0], N_COLS, Z).transpose(1, 2, 0)
    bad = 0
    for i in range(N_ROWS):
        acc = np.zeros(cols.shape[1:], dtype=np.uint8)
        for j in range(N_COLS):
            s = SHIFTS[i][j]
            if s >= 0:
                acc ^= _mul(s, cols[j])
        bad += int(acc.sum())
    return bad


def rate_match(codeword, pl_rate):
    """Full codeword (B, 68Z) -> transmitted bits (B, 7680 + parity)."""
    B = codeword.shape[0]
    nb = PL_PARITY_BLOCKS[pl_rate]
    sys_tx = codeword[:, 2 * Z:K_BITS]                    # blocks 2..21
    par_tx = codeword[:, K_BITS:K_BITS + nb * Z]
    return np.concatenate([sys_tx, par_tx], axis=1)


class Decoder:
    """Iterative decoding on the sub-graph a PL_RATE actually uses.

    Two check rules are available: normalised min-sum (alg="nms", the
    default, what hardware runs) and the exact sum-product tanh rule
    (alg="spa"), kept to measure what the approximation costs; on this
    channel it is about 0.1 dB at the cliff.

    With nb parity blocks transmitted, rows 0..nb-1 are the checks whose
    parities are known; later rows only reference parities that never
    left the transmitter, so they carry no information and are dropped.
    The two punctured systematic blocks enter with zero LLR and are
    recovered by the iteration.
    """

    def __init__(self, pl_rate, iters=40, alpha=0.75, alg="nms"):
        self.nb = PL_PARITY_BLOCKS[pl_rate]
        self.n_cols = K_BLOCKS + self.nb
        self.iters = iters
        self.alpha = alpha
        self.alg = alg                # "nms" | "spa" (exact tanh rule)
        m = (_E_ROW < self.nb) & (_E_COL < self.n_cols)
        self.row = _E_ROW[m]
        self.col = _E_COL[m]
        self.shift = _E_SHIFT[m]
        self.ptr = np.searchsorted(self.row, np.arange(self.nb + 1))

    def llrs_to_cols(self, llr_tx):
        """Transmitted-order LLRs (B, n_tx) -> (n_cols, Z, B) with the
        punctured blocks zeroed."""
        B = llr_tx.shape[0]
        cols = np.zeros((self.n_cols, Z, B), dtype=np.float32)
        n_sys = (K_BLOCKS - 2) * Z
        cols[2:K_BLOCKS] = llr_tx[:, :n_sys].reshape(B, K_BLOCKS - 2, Z).transpose(1, 2, 0)
        cols[K_BLOCKS:] = llr_tx[:, n_sys:].reshape(B, self.nb, Z).transpose(1, 2, 0)
        return cols

    def decode(self, llr_tx):
        """LLRs (B, n_tx), positive = bit 0 -> (info_hat (B, 8448), ok (B,))."""
        Lch = self.llrs_to_cols(np.asarray(llr_tx, dtype=np.float32))
        B = Lch.shape[2]
        E = len(self.row)
        c2v = np.zeros((E, Z, B), dtype=np.float32)
        post = Lch.copy()
        for it in range(self.iters):
            v2c = post[self.col] - c2v                     # extrinsic
            # into the check index domain
            chk = np.empty_like(v2c)
            for e in range(E):
                chk[e] = np.roll(v2c[e], -self.shift[e], axis=0)
            sgn = np.where(chk < 0, -1.0, 1.0).astype(np.float32)
            mag = np.abs(chk)
            new = np.empty_like(chk)
            for r in range(self.nb):
                a, b = self.ptr[r], self.ptr[r + 1]
                if self.alg == "spa":
                    # exact check rule: 2 atanh of the extrinsic product
                    t = np.tanh(np.clip(chk[a:b], -19.0, 19.0) / 2.0)
                    t = np.where(np.abs(t) < 1e-9,
                                 np.sign(t) * 1e-9 + (t == 0) * 1e-9, t)
                    prod = np.prod(t, axis=0)
                    ext = np.clip(prod[None] / t, -0.999999, 0.999999)
                    new[a:b] = 2.0 * np.arctanh(ext)
                    continue
                s_all = np.prod(sgn[a:b], axis=0)
                m = mag[a:b]
                i1 = np.argmin(m, axis=0)
                m1 = np.take_along_axis(m, i1[None], axis=0)[0]
                m_masked = m.copy()
                np.put_along_axis(m_masked, i1[None], np.inf, axis=0)
                m2 = m_masked.min(axis=0)
                mins = np.where(np.arange(b - a)[:, None, None] == i1[None],
                                m2[None], m1[None])
                new[a:b] = (s_all[None] * sgn[a:b]) * (self.alpha * mins)
            for e in range(E):                             # back to var domain
                c2v[e] = np.roll(new[e], self.shift[e], axis=0)
            post = Lch.copy()
            np.add.at(post, self.col, c2v)
            if it % 3 == 2 or it == self.iters - 1:
                hard = (post < 0).astype(np.uint8)
                if self._syndrome_ok(hard).all():
                    break
        hard = (post < 0).astype(np.uint8)
        ok = self._syndrome_ok(hard)
        info = hard[:K_BLOCKS].transpose(2, 0, 1).reshape(B, K_BITS)
        return info, ok

    def _syndrome_ok(self, cols_hard):
        B = cols_hard.shape[2]
        bad = np.zeros(B, dtype=np.int64)
        for r in range(self.nb):
            a, b = self.ptr[r], self.ptr[r + 1]
            acc = np.zeros((Z, B), dtype=np.uint8)
            for e in range(a, b):
                acc ^= np.roll(cols_hard[self.col[e]], -self.shift[e], axis=0)
            bad += acc.sum(axis=0).astype(np.int64)
        return bad == 0

    def decode_hard(self, bits_tx):
        """Hard-input decoding: the same min-sum iteration fed only the
        sign of every channel bit (all magnitudes equal). The gap to
        the soft decoder is then exactly what calibrated LLRs are
        worth, the role iBDD played against Chase-Pyndiah in the fibre
        package. Plain bit-flipping is not usable here: the 768
        punctured bits give it nothing to start from."""
        return self.decode(1.0 - 2.0 * bits_tx.astype(np.float32))
