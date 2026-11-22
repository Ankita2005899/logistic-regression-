"""run_pipeline.py - non-interactive: train, evaluate and print the metrics.
Run:  python run_pipeline.py
"""
from src.model_training import train_and_save
from src.model_evaluation import get_metrics

print("=" * 55)
print("  FULL PIPELINE  (train -> evaluate)")
print("=" * 55)

for key in ["student-mat.csv", "student-por.csv"]:
    r = train_and_save(key)
    m = get_metrics(key, r["pipeline"], r["X_test"], r["y_test"])
    print(f"\nDataset: {key}  | train={r['train_size']}  test={r['test_size']}")
    for k in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "baseline_accuracy"]:
        print(f"  {k:18s}: {m[k]:.4f}")
    print("  Confusion matrix [rows=actual PASS,FAIL; cols=pred PASS,FAIL]:")
    print("  ", m["cm"].tolist())

print("\n[DONE] Pipeline complete.")
