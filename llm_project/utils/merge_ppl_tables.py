import pathlib
import csv

DIR = pathlib.Path("experiments/summary")
files = sorted(DIR.glob("ppl_table_k*.csv"))
out_path = DIR / "ppl_table.csv"

seen = set()
rows = []
for file in files:
    with open(file) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0 and not rows:
                rows.append(row)
            elif i > 0:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

with open(out_path, "w", newline="") as f:
    csv.writer(f).writerows(rows)

print(" Merged:", out_path)
