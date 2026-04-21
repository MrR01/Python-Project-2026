# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
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


# ==============================
# 2. LOAD DATA + HEADERS
# ==============================
columns = ['Age','WorkClass','fnlwgt','Education','Education-Num',
           'Marital-Status','Occupation','Relationship','Race','Gender',
           'Capital-Gain','Capital-Loss','Hrs-per-wk','Country','Income']

df = pd.read_csv("adult.csv", names=columns)

# Replace missing values
df.replace("?", np.nan, inplace=True)

# Target variable encoding
df['Income'] = df['Income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

print(df.head())


# ==============================
# 3. EXPLORATORY ANALYSIS (CLEAN)
# ==============================
def plot_categorical_vs_income(df, column):
    plt.figure()
    sns.barplot(x=column, y="Income", data=df)
    plt.xticks(rotation=45)
    plt.title(f"{column} vs Income")
    plt.tight_layout()
    plt.show()

categorical_cols = ["Gender", "Relationship", "Marital-Status", "WorkClass"]

for col in categorical_cols:
    plot_categorical_vs_income(df, col)


# ==============================
# 4. FEATURE SELECTION (OPTIONAL)
# ==============================
# Keep most features — only drop clearly irrelevant ones
columns_to_drop = ['fnlwgt']  # safe minimal drop
df = df.drop(columns=columns_to_drop)


# ==============================
# 5. TRAIN / TEST SPLIT
# ==============================
X = df.drop("Income", axis=1)
y = df["Income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# 6. PREPROCESSING PIPELINE
# ==============================
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


# ==============================
# 7. SPOT CHECK MODELS (MODERN)
# ==============================
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True)
}

print("\nModel Comparison (ROC-AUC):")

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name}: {scores.mean():.3f}")


# ==============================
# 8. RANDOM FOREST (BASE MODEL)
# ==============================
rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_features=3,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)


# ==============================
# 9. EVALUATION
# ==============================
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==============================
# 10. HYPERPARAMETER TUNING
# ==============================
param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10]
}

tuning_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

search = RandomizedSearchCV(
    tuning_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print("\nBest Parameters:", search.best_params_)
print("Best ROC-AUC:", search.best_score_)