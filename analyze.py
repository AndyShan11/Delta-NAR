"""Print mean+/-std tables from the server grid JSONs."""
import json
import sys

FILES = [("SSSP", "results/grid_sssp.json"), ("BOTTLENECK", "results/grid_bottleneck.json")]
METRICS = ["graph_acc_v", "node_acc_v", "aff_f1"]

for task, f in FILES:
    try:
        d = json.load(open(f))
    except FileNotFoundError:
        print(f"[skip] {f} not found")
        continue
    args = d["args"]
    sizes = args["sizes"]
    seeds = args["seeds"]
    tkey = list(d["agg"].keys())[0]
    models = list(d["agg"][tkey].keys())
    print(f"\n===== {task}  (seeds={seeds}, train_n={args['train_n']}, epochs={args['epochs']}) =====")
    for metric in METRICS:
        print(f"\n-- {metric} (mean+/-std) --")
        print("Model".ljust(11) + "".join(f"n={n}".ljust(16) for n in sizes))
        for m in models:
            row = m.ljust(11)
            for n in sizes:
                mean, std = d["agg"][tkey][m][str(n)][metric]
                row += f"{mean:.3f}+/-{std:.3f}".ljust(16)
            print(row)
