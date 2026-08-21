"""Fine-tune the 200M/100M config around the promising region."""
VOCAB = 32768


def count(d, L, nr, k, rff, sff):
    hkv = max(1, d // 128)
    hq = d // 64
    hd = 64
    embed = VOCAB * d
    attn = d * d + (hkv * hd) * d * 2 + d * d
    shared = 3 * d * sff
    routed = nr * 3 * d * rff
    router = d * nr
    norms = d * 4
    layer = attn + shared + routed + router + norms
    total = embed + L * layer
    per_expert = 3 * d * rff
    inactive = (nr - k) * per_expert * L
    active = total - inactive
    return total, active, inactive


def main():
    print(f"{'d':>5} {'L':>3} {'nr':>4} {'k':>2} {'rff':>6} {'sff':>6} {'total':>10} {'active':>10} {'delta%':>7}")
    best = []
    for d in [768, 896]:
        for L in [8, 10, 12]:
            for nr in [12, 16, 20]:
                for k in [4]:
                    for rff in [512, 640, 768]:
                        for sff in [1536, 2048, 2304]:
                            total, active, inactive = count(d, L, nr, k, rff, sff)
                            score = abs(total - 200e6) / 1e6 + abs(active - 100e6) / 1e6
                            best.append((score, d, L, nr, k, rff, sff, total, active))
    best.sort()
    print("\nTop 12 tuned configs:")
    for score, d, L, nr, k, rff, sff, total, active in best[:12]:
        dt = (total - 200e6) / 200e6 * 100
        da = (active - 100e6) / 100e6 * 100
        print(f"  d={d} L={L} nr={nr} k={k} rff={rff} sff={sff} | total={total/1e6:.1f}M "
              f"active={active/1e6:.1f}M (dt={dt:+.1f}% da={da:+.1f}%)")


if __name__ == "__main__":
    main()
