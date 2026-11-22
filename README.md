# Student Performance Prediction – Logistic Regression (CIE 2, Activity 2)

**Course:** Regression and Classification Algorithms | T.Y. B.Tech AI & DS | Zeal College of Engineering and Research
**Faculty GitHub ID:** [https://github.com/complexiota](https://github.com/complexiota)

## Activity
Binary classification with Logistic Regression: predict whether a student will **FAIL** (`G3 < 10`, positive class = student at risk) or PASS, and evaluate it as a professional data scientist would.

## Dataset
UCI Machine Learning Repository – *Student Performance* (P. Cortez & A. Silva, 2008), Mathematics file `student-mat.csv`: 395 rows, 33 columns. Source: https://archive.ics.uci.edu/dataset/320/student+performance
`G1`, `G2`, `G3` (leakage) and `Dalc`, `Walc` (sensitive) are not used as features.

## Work completed
- Problem framing, data inspection, class-balance plot
- Stratified 80:20 split (seed 42), pipeline with imputation, scaling and one-hot encoding (fitted on training data only)
- Sigmoid plot, logistic regression coefficients and odds ratios
- Predicted probabilities and labels, labelled confusion matrix (TP/FP/FN/TN)
- Accuracy, precision, recall, specificity, F1, classification report, ROC-AUC, precision–recall curve
- Majority-class baseline comparison, threshold table (0.3 / 0.5 / 0.7), train-vs-test and cross-validation check
- Decision-behaviour plot, one new prediction, decision log, conclusion and limitations
- Bonus: Streamlit dashboard (`app.py`) with a colour-themed input form, live prediction, risk gauge and "why this prediction" chart, plus a terminal app (`main.py`)

## Key results (test set, threshold 0.5, positive = FAIL)
| Metric | Value |
|---|---|
| Accuracy | 0.734 (baseline always-PASS: 0.671) |
| Precision | 0.667 |
| Recall | 0.385 (baseline: 0.000) |
| ROC-AUC | 0.713 (5-fold CV mean: 0.633) |

Lowering the threshold to 0.3 raises recall to 0.654 (precision 0.459). Because a missed at-risk student costs more than an unnecessary support offer, 0.3 is recommended. The model is a weak screening aid and must not be used to make decisions about students on its own.

## Folder structure
```
Logistic_Regression_Activity2/
├── notebooks/Activity2_Logistic_Regression.ipynb   # main report (executed)
├── dataset/            # student-mat.csv, student-por.csv
├── src/                # preprocessing, training, evaluation, prediction, plots
├── visualizations/     # saved figures
├── models/             # trained models are created here (not committed)
├── .streamlit/config.toml  # app colour theme
├── app.py              # Streamlit dashboard
├── main.py             # terminal menu app
├── run_pipeline.py     # train + evaluate from the command line
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python run_pipeline.py          # train + evaluate
streamlit run app.py            # dashboard (trains the model automatically on first use)
python main.py                  # terminal menu (choose 2 = Train first)
jupyter notebook notebooks/Activity2_Logistic_Regression.ipynb
```
