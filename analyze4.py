"""Summarize v4 batch (#12 extreme size, #14 seeds->10)."""
import json
import os

R = "results"


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


print("### #14 main grid, 10 seeds -- graph_acc_v / node_acc_v")
s14 = merge(["SEED14_sssp.json", "SEED14_bot.json"])
for task in ["sssp", "bottleneck"]:
    md = s14.get((task, "DeltaNAR"), {})
    print(f"-- {task} DeltaNAR --")
    for n in [16, 32, 64, 128, 256]:
        if n in md:
            g = md[n]["graph_acc_v"]
            no = md[n]["node_acc_v"]
            print(f"  n={n:>4}: graph={g[0]:.3f}+-{g[1]:.3f}  node={no[0]:.3f}+-{no[1]:.3f}")

print("\n### #12 extreme size (steps=32) -- node_acc_v / graph_acc_v")
e12 = merge(["EXT12_sssp.json", "EXT12_bot.json"])
for task in ["sssp", "bottleneck"]:
    for m in ["DeltaNAR", "Recompute"]:
        if (task, m) in e12:
            d = e12[(task, m)]
            for n in [512, 1024]:
                if n in d:
                    print(f"  {task:10s} {m:10s} n={n:>4}: "
                          f"node={d[n]['node_acc_v'][0]:.3f}+-{d[n]['node_acc_v'][1]:.3f}  "
                          f"graph={d[n]['graph_acc_v'][0]:.3f}")
