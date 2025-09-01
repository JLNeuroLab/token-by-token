import csv
import pathlib
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_PATH = pathlib.Path("experiments/summary/ppl_table.csv")
OUTDIR = CSV_PATH.parent
data = []  # (model, k, ppl)

with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        data.append((row["model"], int(row["k"]), float(row["val_perplexity"])))

# Plot 1: PPL vs k per model
by_model = defaultdict(list)
for m, k, ppl in data:
    by_model[m].append((k, ppl))
for m in by_model:
    by_model[m].sort(key=lambda x: x[0])

plt.figure()
for m, pts in by_model.items():
    ks, ppls = zip(*pts)
    plt.plot(ks, ppls, marker="o", label=m)
plt.xlabel("BPE merges (k)")
plt.ylabel("Validation perplexity")
plt.title("Perplexity vs. BPE merges (k)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
p1 = OUTDIR / "fig_ppl_vs_k_by_model.png"
plt.savefig(p1, dpi=180)

# Plot 2: Best model (lowest ppl) at its best k
best = {}
for m, pts in by_model.items():
    best[m] = min(pts, key=lambda x: x[1])  # (k, ppl)

labels = list(best.keys())
vals = [best[m][1] for m in labels]
ks = [best[m][0] for m in labels]

plt.figure()
plt.bar(labels, vals)
for i, (v, k) in enumerate(zip(vals, ks)):
    plt.text(i, v, f"k={k}\n{v:.1f}", ha="center", va="bottom")
plt.ylabel("Validation perplexity")
plt.title("Best PPL per model (annotated with k)")
plt.tight_layout()
p2 = OUTDIR / "fig_best_model_bar.png"
plt.savefig(p2, dpi=180)

print(f" Saved:\n - {p1}\n - {p2}")
