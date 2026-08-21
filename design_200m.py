"""Analytic param search for 200M-total / 100M-active MoE config.

Param formulas for the research_hybrid architecture:
  embedding (tied with LM head) = vocab * d
  per layer:
    attn Q/KV/O: d*d (Q) + (hkv*hd)*d*2 (KV) + d*d (O)   [GQA, bias-less]
    shared expert: 3 * d * shared_d_ff
    routed experts: n_routed * (3 * d * routed_d_ff)
    router: d * n_routed
    norms ~ d (negligible, include)
  total = embed + L*(attn + shared + routed + router + norms)
  active = total - (n_routed - top_k) * (3*d*routed_d_ff) * L
"""
VOCAB = 32768


def count(d, L, nr, k, rff, sff=None, hkv_ratio=128, nq_ratio=64):
    hkv = max(1, d // hkv_ratio)
    hq = d // nq_ratio
    hd = 64
    sff = sff or 2 * d
    embed = VOCAB * d
    attn = d * d + (hkv * hd) * d * 2 + d * d
    shared = 3 * d * sff
    routed = nr * 3 * d * rff
    router = d * nr
    norms = d * 4  # RMSNorm weights per block (approx)
    layer = attn + shared + routed + router + norms
    total = embed + L * layer
    per_expert = 3 * d * rff
    inactive = (nr - k) * per_expert * L
    active = total - inactive
    return total, active, inactive


def main():
    print(f"{'d':>5} {'L':>3} {'nr':>4} {'k':>2} {'rff':>6} {'sff':>6} {'total':>10} {'active':>10} {'inactive':>10}")
    best = []
    near = []
    for d in [768, 896, 1024, 1152, 1280]:
        for L in [8, 10, 12, 14, 16, 20]:
            for nr in [8, 12, 16, 24, 32, 48]:
                for k in [4, 6, 8]:
                    for rff in [512, 768, 896, 1024, 1280, 1536]:
                        total, active, inactive = count(d, L, nr, k, rff)
                        if 180e6 <= total <= 220e6 and 90e6 <= active <= 110e6:
                            score = abs(total - 200e6) + abs(active - 100e6)
                            best.append((score, d, L, nr, k, rff, 2 * d, total, active, inactive))
                        # track closest near-misses for diagnostics
                        if 150e6 <= total <= 250e6 and 80e6 <= active <= 120e6:
                            score = abs(total - 200e6) + abs(active - 100e6)
                            near.append((score, d, L, nr, k, rff, 2 * d, total, active, inactive))
    best.sort()
    near.sort()
    print(f"\nTop {min(15, len(best))} configs near 200M/100M:")
    for score, d, L, nr, k, rff, sff, total, active, inactive in best[:15]:
        print(f"  d={d} L={L} nr={nr} k={k} rff={rff} sff={sff} | "
              f"total={total/1e6:.1f}M active={active/1e6:.1f}M inactive={inactive/1e6:.1f}M")
    if not best:
        print("(none exact — closest near-misses in range 150-250M/80-120M:)")
        for score, d, L, nr, k, rff, sff, total, active, inactive in near[:15]:
            print(f"  d={d} L={L} nr={nr} k={k} rff={rff} sff={sff} | "
                  f"total={total/1e6:.1f}M active={active/1e6:.1f}M inactive={inactive/1e6:.1f}M")


if __name__ == "__main__":
    main()
