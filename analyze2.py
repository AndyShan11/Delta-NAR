"""Summarize the v2 server batch (results/*.json)."""
import json
import os

R = "results"
ORDER = ["DeltaNAR", "Naive-d", "Recomp+gate", "Triplet", "Recompute"]


def L(f):
    p = os.path.join(R, f)
    return json.load(open(p)) if os.path.exists(p) else None


def merge(files):
    out = {}
    for f in files:
        d = L(f)
        if not d:
            continue
        tk = list(d["agg"].keys())[0]
        for m, nd in d["agg"][tk].items():
            out.setdefault(m, {})
            for n, md in nd.items():
                out[m][int(n)] = md
    return out


def tbl(title, data, metric, sizes):
    print(f"\n### {title} -- {metric}")
    print("model".ljust(12) + "".join(f"n={n}".ljust(14) for n in sizes))
    for m in ORDER:
        if m not in data:
            continue
        row = m.ljust(12)
        for n in sizes:
            md = data[m].get(n)
            row += (f"{md[metric][0]:.3f}+-{md[metric][1]:.3f}".ljust(14)) if md and metric in md else "-".ljust(14)
        print(row)


sd = merge(["A_sssp.json", "A_sssp_triplet.json"])
tbl("SSSP main grid", sd, "graph_acc_v", [16, 32, 64, 128, 256])
tbl("SSSP main grid", sd, "node_acc_v", [16, 32, 64, 128, 256])
bd = merge(["A_bot.json", "A_bot_triplet.json"])
tbl("BOTTLENECK main grid", bd, "node_acc_v", [16, 32, 64, 128, 256])
tbl("BOTTLENECK main grid", bd, "graph_acc_v", [16, 32, 64, 128, 256])

er = merge(["A_sssp.json"])
ba = merge(["B_sssp_ba.json"])
gr = merge(["B_sssp_grid.json"])
print("\n### TOPOLOGY OOD @ n=64 (train ER) -- node_acc_v")
print("model".ljust(12) + "ER".ljust(12) + "BA".ljust(12) + "grid".ljust(12))
for m in ORDER:
    if not any(m in s for s in (er, ba, gr)):
        continue
    row = m.ljust(12)
    for src in (er, ba, gr):
        md = src.get(m, {}).get(64)
        row += (f"{md['node_acc_v'][0]:.3f}".ljust(12)) if md else "-".ljust(12)
    print(row)

whi = merge(["C_sssp_whi100.json"])
hv = merge(["C_sssp_heavy.json"])
print("\n### WEIGHT OOD @ n=64 (train w in [1,10]) -- node_acc_v")
print("model".ljust(12) + "normal".ljust(12) + "w<=100".ljust(12) + "heavy".ljust(12))
for m in ["DeltaNAR", "Recompute"]:
    row = m.ljust(12)
    for src in (er, whi, hv):
        md = src.get(m, {}).get(64)
        row += (f"{md['node_acc_v'][0]:.3f}".ljust(12)) if md else "-".ljust(12)
    print(row)

e1 = merge(["E_sssp_512.json"])
e2 = merge(["E_bot_512.json"])
print("\n### EXTREME n=512 -- node_acc_v / graph_acc_v")
for tag, src in [("SSSP", e1), ("BOTTLENECK", e2)]:
    for m in ["DeltaNAR", "Recompute"]:
        md = src.get(m, {}).get(512)
        if md:
            print(f"{tag:10s} {m:10s} node {md['node_acc_v'][0]:.3f}+-{md['node_acc_v'][1]:.3f}  graph {md['graph_acc_v'][0]:.3f}")
