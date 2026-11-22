import os
import joblib
from .data_preprocessing import prepare_data, build_pipeline, MODELS_DIR, MODEL_PATH

def get_model_path(dataset_key="student-mat.csv"):
    clean_name = "student_performance_model"
    if "por" in dataset_key.lower():
        clean_name += "_por"
    elif "combined" in dataset_key.lower():
        clean_name += "_combined"
    return os.path.join(MODELS_DIR, f"{clean_name}.pkl")

def train_and_save(dataset_key="student-mat.csv") -> dict:
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(dataset_key)
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    save_path = get_model_path(dataset_key)
    joblib.dump(pipeline, save_path)
    
    # Also save as default model if it's math
    if dataset_key == "student-mat.csv":
        joblib.dump(pipeline, MODEL_PATH)

    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_cols": feature_cols,
        "model_path": save_path,
        "dataset_key": dataset_key,
    }

def load_pipeline(dataset_key="student-mat.csv"):
    path = get_model_path(dataset_key)
    if os.path.exists(path):
        return joblib.load(path)
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None
