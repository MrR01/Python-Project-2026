# =========================================
# ADULT INCOME PROJECT - 2026 VERSION
# Automated & Workflow-Optimised
# =========================================

# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import joblib

# ==============================
# 2. CONFIGURATION (EASY TO MODIFY)
# ==============================
DATA_PATH = "/kaggle/input/datasets/organizations/uciml/adult-census-income/adult.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_PATH = "outputs/"
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ==============================
# 3. LOAD & CLEAN DATA
# ==============================
def load_data(path):
    """
    Loads dataset and applies initial cleaning
    """
    columns = ['Age','WorkClass','fnlwgt','Education','Education-Num',
               'Marital-Status','Occupation','Relationship','Race','Gender',
               'Capital-Gain','Capital-Loss','Hrs-per-wk','Country','Income']

    df = pd.read_csv(path, names=columns, header=0)

    # Replace missing values represented as '?'
    df.replace("?", np.nan, inplace=True)
    #print(df.isnull().sum())
    df["WorkClass"] = df["WorkClass"].fillna("Unknown")
    df["Occupation"] = df["Occupation"].fillna("Unknown")

    # Convert target variable to binary (0/1)
    df['Income'] = df['Income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    return df
df = load_data(DATA_PATH)
df.head()

# ==============================
# 4. EXPLORATORY ANALYSIS
# ==============================
def explore_data(df):
    """
    Generate visual insights with spacing + save plots
    """

    categorical_cols = ["Gender", "Relationship", "Marital-Status", "WorkClass"]

    for col in categorical_cols:
        plt.figure(figsize=(8, 5))  # better size

        sns.barplot(
            x=col,
            y="Income",
            hue=col,
            data=df,
            palette="Set2",
            legend=False
        )

        plt.xticks(rotation=45)
        plt.title(f"{col} vs Income")

        plt.tight_layout(pad=2)  # spacing

        # SAVE IMAGE
        safe_col = col.replace(" ", "_")
        plt.savefig(f"{OUTPUT_PATH}{safe_col}_vs_income.png", dpi=300, bbox_inches="tight")

        plt.show()

# ==============================
# 5. FEATURE SELECTION
# ==============================
def select_features(df):
    """
    Drop clearly irrelevant columns (minimal manual intervention)
    """
    df = df.drop(columns=['fnlwgt'])  # safe to remove
    return df

# ==============================
# 6. BUILD PREPROCESSING PIPELINE
# ==============================
def build_preprocessor(X):
    """
    Creates preprocessing pipeline for numeric and categorical data
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor



# ==============================
# 7. MODEL COMPARISON
# ==============================
def compare_models(X_train, y_train, preprocessor):

    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True)
    }

    results_list = []  # store results

    print("\nModel Comparison (ROC-AUC):")

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        )

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{name}: {mean_score:.3f} (+/- {std_score:.3f})")

        # save results
        results_list.append({
            "Model": name,
            "Mean ROC-AUC": mean_score,
            "Std Dev": std_score
        })

    # convert to DataFrame and save
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_PATH + "model_comparison.csv", index=False)

# ==============================
# 8. TRAIN MODEL
# ==============================
def train_model(X_train, y_train, preprocessor):
    """
    Train Random Forest model with defined parameters
    """
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_features=3,
            random_state=RANDOM_STATE
        ))
    ])

    model.fit(X_train, y_train)
    return model

# ==============================
# 9. EVALUATE MODEL
# ==============================
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print("ROC AUC:", roc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    # SAVE metrics
    metrics_df = pd.DataFrame({
        "Metric": ["ROC-AUC", "F1 Score", "Precision", "Recall"],
        "Value": [roc, f1, precision, recall]
    })

    metrics_df.to_csv(OUTPUT_PATH + "evaluation_metrics.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    #  SAVE IMAGE
    plt.savefig(OUTPUT_PATH + "confusion_matrix.png", dpi=300, bbox_inches="tight")

    plt.show()


# ==============================
# 10. HYPERPARAMETER TUNING
# ==============================
def tune_model(X_train, y_train, preprocessor):
    """
    Perform RandomizedSearchCV to optimise Random Forest
    """
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:", search.best_params_)
    print("Best ROC-AUC:", search.best_score_)

    return search.best_estimator_

# ==============================
# 11. SAVE MODEL
# ==============================
def save_model(model):
    """
    Save trained model for reuse
    """
    joblib.dump(model, OUTPUT_PATH + "model.pkl")

# ==============================
# 12. MAIN WORKFLOW (AUTOMATED)
# ==============================
def main():
    """
    Runs full pipeline automatically
    """

    # Load & clean
    df = load_data(DATA_PATH)

    # Optional EDA
    explore_data(df)

    # Feature selection
    df = select_features(df)

    # Split data
    X = df.drop("Income", axis=1)
    y = df["Income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Build preprocessing
    preprocessor = build_preprocessor(X)

    # Compare models
    compare_models(X_train, y_train, preprocessor)

    # Train base model
    model = train_model(X_train, y_train, preprocessor)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Tune model
    best_model = tune_model(X_train, y_train, preprocessor)

    # Save best model
    save_model(best_model)


# ==============================
# RUN SCRIPT
# ==============================
if __name__ == "__main__":
    main()

