"""End to end: payload bits -> OCT modem frames -> optical waveform ->
vacuum channel -> preamplified receiver -> timing recovery -> frame
sync -> decoders -> BER.

The stream is simulated continuously, frames back to back as the
standard requires. The first frame is the acquisition frame: the
timing loop and the level estimators settle during it and it is
excluded from every count, which is how a real terminal behaves during
link stand-up. Pre-FEC BER is counted on the transmitted coded bits,
post-FEC on the 8416 payload bits per frame, and the per-frame CRC-32
gives the standard's own definition of a good frame.
"""

import numpy as np

from . import channel, frontend, rxdsp
from .config import SimConfig
from .fec import ldpc
from .framing import (HEADER_CODED, PAYLOAD_BITS, build_frames, frame_bits,
                      parse_header, split_frame)


def _to_chips(line, encoding):
    if encoding == "nrz":
        return line
    b = line.reshape(line.shape[0], -1, 1)
    chips = np.concatenate([b, 1 - b], axis=2)      # 1 -> {1,0}, 0 -> {0,1}
    return chips.reshape(line.shape[0], -1)


def run_link(cfg: SimConfig = None, seed=1):
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    F = cfg.n_frames
    nb = frame_bits(cfg.pl_rate)
    n_chips = nb if cfg.encoding == "nrz" else 2 * nb

    payload = rng.integers(0, 2, (F, PAYLOAD_BITS), dtype=np.uint8)
    line, body_ref = build_frames(payload, cfg.pl_rate, cfg.encoding)
    chips = _to_chips(line, cfg.encoding).reshape(-1)

    power = frontend.tx_waveform(chips, cfg, rng)

    budget = channel.link_budget(cfg)
    bit_rate = cfg.baud if cfg.encoding == "nrz" else cfg.baud / 2
    if cfg.rx_photons_per_bit > 0:
        ph_bit = cfg.rx_photons_per_bit
    else:
        ph_bit = budget["photons_per_bit"]
    p_ph_s = ph_bit * bit_rate

    frame_t = n_chips / cfg.baud
    if cfg.include_impairments:
        gains = channel.pointing_gains(cfg, F, frame_t, rng)
    else:
        gains = np.ones(F)
    scale = np.repeat(gains, n_chips * cfg.sps)[: power.size]
    power = power * scale * p_ph_s

    wave = frontend.receiver(power, cfg, rng)
    ppm = cfg.clock_ppm if cfg.include_impairments else 0.0
    tau0 = rng.uniform(-0.5, 0.5) if cfg.include_impairments else 0.0
    x2 = frontend.adc(wave, cfg, rng, ppm, tau0)

    y, f_trace = rxdsp.gardner(x2, cfg)
    from .framing import PREAMBLE
    sync_ref = _to_chips(PREAMBLE[None], cfg.encoding)[0]
    off = rxdsp.frame_sync(y, n_chips, sync_ref)

    n_full = (y.size - off) // n_chips
    first = 1 if n_full > 1 else 0                  # acquisition frame out
    idx = off + np.arange(first, n_full) * n_chips
    frames_y = np.stack([y[a:a + n_chips] for a in idx])

    if cfg.encoding == "manchester":
        pairs = frames_y.reshape(frames_y.shape[0], -1, 2)
        frames_y = 0.5 * (pairs[:, :, 0] - pairs[:, :, 1]) + \
            0.5 * (pairs[:, :, 0] + pairs[:, :, 1]).mean()

    # per-frame level tracking (pointing fades frame by frame), then LLRs
    fr_llr = np.empty_like(frames_y)
    for i, fy in enumerate(frames_y):
        mu0, mu1, s0, s1 = rxdsp.estimate_levels(fy[64:])
        if cfg.ffe_taps > 1:
            fy = rxdsp.ffe(fy, mu0, mu1, cfg.ffe_taps)
            mu0, mu1, s0, s1 = rxdsp.estimate_levels(fy[64:])
        fr_llr[i] = rxdsp.llrs(fy, mu0, mu1, s0, s1)

    hdr_llr, body_llr = split_frame(fr_llr[:, 64:], cfg.pl_rate)
    _, hdr_ok, hdr_pl, hdr_idx = parse_header(hdr_llr)

    # frame identity comes from the headers: any frame with a good CRC
    # anchors the synchronous stream, the rest follow by position
    pos = np.arange(len(idx))
    if hdr_ok.any():
        anchor = int(np.median(hdr_idx[hdr_ok] - pos[hdr_ok]))
    else:
        anchor = first
    which = anchor + pos
    keep = (which >= 0) & (which < F)
    which, body_llr, hdr_ok = which[keep], body_llr[keep], hdr_ok[keep]
    ref = body_ref[which]
    pay = payload[which]

    hard = (body_llr < 0).astype(np.uint8)
    pre_err = int((hard != ref).sum())
    pre_bits = int(ref.size)

    res = {
        "photons_per_bit": ph_bit,
        "budget": budget,
        "pointing_gain_db": 10 * np.log10(np.maximum(gains[first:], 1e-12)),
        "clock_ppm_est": float(f_trace[-1] * 1e6) if f_trace.size else 0.0,
        "header_ok": int(hdr_ok.sum()),
        "n_frames_counted": len(which),
        "pre_fec_ber": pre_err / max(pre_bits, 1),
        "pre_fec_errors": pre_err,
        "pre_fec_bits": pre_bits,
    }

    if cfg.pl_rate == 0:
        post = hard[:, :PAYLOAD_BITS]
        res["post_fec"] = {"sd": _count(post, pay), "hd": _count(post, pay)}
        return res

    dec = ldpc.Decoder(cfg.pl_rate, iters=cfg.ldpc_iters, alg=cfg.ldpc_alg)
    info_sd, ok_sd = dec.decode(body_llr)
    info_hd, ok_hd = dec.decode_hard(hard)
    res["post_fec"] = {
        "sd": _count(info_sd[:, :PAYLOAD_BITS], pay, ok_sd),
        "hd": _count(info_hd[:, :PAYLOAD_BITS], pay, ok_hd),
    }
    return res


def _count(got, want, ok=None):
    err = int((got != want).sum())
    out = {"errors": err, "bits": int(want.size),
           "ber": err / max(want.size, 1),
           "frame_errors": int((got != want).any(axis=1).sum()),
           "frames": int(want.shape[0])}
    if ok is not None:
        out["decoder_converged"] = int(np.asarray(ok).sum())
    return out
