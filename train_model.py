import pandas as pd
from feature_extractor import extract_features
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv("Data/phishing_url.csv")

feature_list = df['url'].apply(extract_features)
X = pd.DataFrame(feature_list.tolist())
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    class_weight={0: 1, 1: 1.5},
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.4).astype(int)

print(classification_report(y_test, y_pred))



joblib.dump(model, "models/phishradar_model.joblib")
joblib.dump(X.columns.tolist(), "models/feature_columns.joblib")