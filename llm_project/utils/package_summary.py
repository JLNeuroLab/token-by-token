import pathlib
import shutil

BASE = pathlib.Path("experiments")
DST = BASE / "summary"
DST.mkdir(parents=True, exist_ok=True)

copies = [
    "experiments/summary/ppl_table.csv",
    "experiments/summary/fig_ppl_vs_k_by_model.png",
    "experiments/summary/fig_best_model_bar.png",
    "experiments/plots/gpt/loss_curve.png",
    "experiments/plots/gpt/val_perplexity.png",
    "experiments/saved_models/neural_ngram/perplexity_curve.png",
    "experiments/saved_models/neural_ngram/vocabulary_growth.png",
    "experiments/saved_models/ngram/lambda_perplexity.png",
    "experiments/saved_models/ngram/vocabulary_growth.png",
    "experiments/plots/ngram_comparison/lambda_perplexity.png",
]

for p in copies:
    src = pathlib.Path(p)
    if src.exists():
        shutil.copy2(src, DST / src.name)

# Copy generations
gen_src = BASE / "summary" / "generations"
if gen_src.exists():
    dest = DST / "generations"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(gen_src, dest)

print("Packaged to:", DST.resolve())
