"""
main.py - CLI Entrypoint for Student Performance Prediction
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import load_raw_data, create_target, get_feature_types, MODEL_PATH
from src.model_training import train_and_save, load_pipeline
from src.model_evaluation import get_metrics
from src.prediction import predict_student

def print_menu():
    print("\n" + "="*50)
    print(" STUDENT PERFORMANCE PREDICTION (LOGISTIC REGRESSION)")
    print("="*50)
    print("1. Dataset Overview")
    print("2. Train Model")
    print("3. Evaluate Model")
    print("4. Predict Student Result")
    print("5. Launch Streamlit Visual Dashboard")
    print("6. Exit")
    print("="*50)

def main():
    while True:
        print_menu()
        choice = input("Enter choice (1-6): ").strip()
        if choice == '1':
            df = load_raw_data()
            print("\nDataset Shape:", df.shape)
            print("\nColumns:", df.columns.tolist())
            print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
            print("\nSummary Statistics:\n", df.describe().round(2))
        elif choice == '2':
            print("\nTraining Logistic Regression model...")
            res = train_and_save()
            print(f"[SUCCESS] Model trained & saved to: {res['model_path']}")
            print(f"Train samples: {res['train_size']} | Test samples: {res['test_size']}")
        elif choice == '3':
            m = get_metrics()
            if not m:
                print("[ERROR] Please train model first (Option 2).")
                continue
            print("\n--- Model Evaluation ---")
            print(f"Accuracy  : {m['accuracy']*100:.2f}%")
            print(f"Precision : {m['precision']:.4f}")
            print(f"Recall    : {m['recall']:.4f}")
            print(f"F1-Score  : {m['f1']:.4f}")
            print(f"ROC-AUC   : {m['roc_auc']:.4f}")
            print("\nConfusion Matrix:\n", m['cm'])
        elif choice == '4':
            pipeline = load_pipeline()
            if not pipeline:
                print("[ERROR] Please train model first (Option 2).")
                continue
            raw = load_raw_data()
            df = create_target(raw)
            feat_df = df.drop(columns=['result'])
            num_cols, cat_cols = get_feature_types(feat_df)
            user_input = {}
            print("\nEnter sample values (press enter to use dataset median/mode default):")
            for col in cat_cols:
                mode_val = feat_df[col].mode()[0]
                val = input(f" {col} [{mode_val}]: ").strip()
                user_input[col] = val if val else mode_val
            for col in num_cols:
                med_val = int(feat_df[col].median())
                val = input(f" {col} [{med_val}]: ").strip()
                user_input[col] = float(val) if val else float(med_val)
            res = predict_student(user_input)
            print("\n--- PREDICTION RESULT ---")
            print(f"Predicted Outcome : {res['prediction']}")
            print(f"PASS Probability  : {res['pass_prob']}%")
            print(f"FAIL Probability  : {res['fail_prob']}%")
        elif choice == '5':
            print("\nStarting Streamlit Visual Dashboard...")
            os.system("streamlit run app.py")
        elif choice == '6':
            print("\nExiting. Goodbye!")
            break
        else:
            print("[!] Invalid option. Please enter 1-6.")

if __name__ == '__main__':
    main()
