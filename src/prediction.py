import pandas as pd
from .model_training import load_pipeline

def predict_student(input_dict: dict, dataset_key="student-mat.csv") -> dict:
    pipeline = load_pipeline(dataset_key)
    if pipeline is None:
        return {"error": "Model not trained yet. Please train the model first."}

    input_df = pd.DataFrame([input_dict])
    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    classes = list(pipeline.named_steps["model"].classes_)
    pass_p = float(proba[classes.index("PASS")] * 100)
    fail_p = float(proba[classes.index("FAIL")] * 100)

    return {
        "prediction": pred,
        "pass_prob": round(pass_p, 2),
        "fail_prob": round(fail_p, 2),
    }


def explain_prediction(input_dict: dict, dataset_key="student-mat.csv", top_n=8):
    """Per-feature effect on the log-odds of FAIL (+ = towards FAIL, - = towards PASS).
    Effect = coefficient x transformed value, summed back to the original feature."""
    import numpy as np
    pipeline = load_pipeline(dataset_key)
    if pipeline is None:
        return pd.DataFrame(columns=["feature", "effect"])
    pre = pipeline.named_steps["preprocessor"]
    lr = pipeline.named_steps["model"]
    cat_cols = list(pre.transformers_[1][2])
    X = pre.transform(pd.DataFrame([input_dict]))[0]
    names = pre.get_feature_names_out()
    classes = list(lr.classes_)
    sign = -1.0 if classes[1] == "PASS" else 1.0     # coef is for classes_[1]
    effects = {}
    for n, v, c in zip(names, X, lr.coef_[0]):
        raw = n.split("__", 1)[1]
        feat = raw
        if n.startswith("cat__"):
            matches = [col for col in cat_cols if raw.startswith(col + "_")]
            feat = max(matches, key=len) if matches else raw
        effects[feat] = effects.get(feat, 0.0) + sign * c * v
    out = pd.DataFrame({"feature": list(effects), "effect": list(effects.values())})
    out = out.reindex(out["effect"].abs().sort_values(ascending=False).index)
    return out.head(top_n).reset_index(drop=True)
