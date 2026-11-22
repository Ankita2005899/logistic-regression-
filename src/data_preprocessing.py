import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "student-mat.csv")
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "student_performance_model.pkl")

EXCLUDE_COLS = ["G1", "G2", "G3", "Dalc", "Walc"]
TARGET_COL = "result"

def get_available_datasets():
    return {
        "Mathematics (student-mat.csv)": "student-mat.csv",
        "Portuguese Language (student-por.csv)": "student-por.csv",
        "Both Subjects Combined (Math + Portuguese)": "combined"
    }

def read_csv_robust(file_path: str) -> pd.DataFrame:
    """Robustly reads a CSV file checking both semicolon and comma delimiters."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    try:
        # First attempt with semicolon
        df = pd.read_csv(file_path, sep=';')
        if len(df.columns) > 1 and "G3" in df.columns:
            return df
    except Exception:
        pass
    # Fallback with python engine autodetection
    return pd.read_csv(file_path, sep=None, engine='python')

def load_raw_data(dataset_key="student-mat.csv") -> pd.DataFrame:
    mat_path = os.path.join(DATASET_DIR, "student-mat.csv")
    por_path = os.path.join(DATASET_DIR, "student-por.csv")

    if dataset_key == "combined":
        df_m = read_csv_robust(mat_path)
        df_m["subject"] = "Mathematics"
        df_p = read_csv_robust(por_path)
        df_p["subject"] = "Portuguese"
        df = pd.concat([df_m, df_p], ignore_index=True)
        return df
    elif "por" in dataset_key.lower():
        return read_csv_robust(por_path)
    else:
        return read_csv_robust(mat_path)

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "G3" not in df.columns:
        raise KeyError(f"'G3' column not found in DataFrame. Available columns: {df.columns.tolist()}")
    df[TARGET_COL] = df["G3"].apply(lambda g: "PASS" if g >= 10 else "FAIL")
    # Drop leakage columns
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    return df

def get_feature_types(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    return num_cols, cat_cols

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols, cat_cols = get_feature_types(X)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
    ])

def prepare_data(dataset_key="student-mat.csv"):
    raw = load_raw_data(dataset_key)
    df = create_target(raw)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()
