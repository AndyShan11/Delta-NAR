"""Summarize v3 batch (#7-#11)."""
import json
import os

R = "results"
SIZES = [16, 32, 64, 128]


def L(f):
    p = os.path.join(R, f)
    return json.load(open(p)) if os.path.exists(p) else None


def merge(files):
    out = {}
    for f in files:
        d = L(f)
        if not d:
            continue
        for tk in d["agg"]:
            for m, nd in d["agg"][tk].items():
                out.setdefault((tk, m), {})
                for n, md in nd.items():
                    out[(tk, m)][int(n)] = md
    return out


print("### #7 component ablation -- node_acc_v")
abl = merge(["ABL7.json"])
for task in ["sssp", "bottleneck"]:
    print(f"-- {task} --")
    print("model".ljust(11) + "".join(f"n={n}".ljust(13) for n in SIZES))
    for m in ["DeltaNAR", "no-copy", "min-only", "Naive-d", "Recompute"]:
        if (task, m) in abl:
            row = m.ljust(11)
            for n in SIZES:
                md = abl[(task, m)].get(n)
                row += (f"{md['node_acc_v'][0]:.3f}".ljust(13)) if md else "-".ljust(13)
            print(row)

print("\n### #9 Attn baseline -- node_acc_v")
at = merge(["BASE9_attn.json"])
for task in ["sssp", "bottleneck"]:
    if (task, "Attn") in at:
        d = at[(task, "Attn")]
        print(f"  {task}: " + "  ".join(f"n{n}={d[n]['node_acc_v'][0]:.3f}" for n in SIZES if n in d))

print("\n### #10 failure-OOD (test=edge_delete) -- node_acc_v")
f10 = merge(["OOD10_failure.json"])
for task in ["sssp", "bottleneck"]:
    for m in ["DeltaNAR", "Recompute"]:
        if (task, m) in f10:
            d = f10[(task, m)]
            print(f"  {task:10s} {m:10s}: " + "  ".join(f"n{n}={d[n]['node_acc_v'][0]:.3f}" for n in [16, 64] if n in d))

print("\n### #11 SSSP->bottleneck transfer -- node_acc_v")
t11 = L("TRANS11.json")
if t11:
    for seed, d in t11.items():
        zs, fs = d["zero_shot"], d["few_shot"]
        print(f"  seed{seed}: zero-shot n16={zs['16']['node_acc_v']:.3f}/n64={zs['64']['node_acc_v']:.3f}"
              f"  ->  few-shot n16={fs['16']['node_acc_v']:.3f}/n64={fs['64']['node_acc_v']:.3f}")
