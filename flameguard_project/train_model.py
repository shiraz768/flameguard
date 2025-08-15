import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pickle
import os

DATA_FILE = 'forestfires.csv'

def load_real_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset '{DATA_FILE}' not found.")
    df = pd.read_csv(DATA_FILE)

  
    if 'fire' in df.columns:
        df['fire'] = df['fire'].astype(int)
    elif 'area' in df.columns:
        df['fire'] = (df['area'] > 0).astype(int)

    df['oxygen'] = 21 - (df['RH'] / 100) * 6

    return df

def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        temp = np.random.uniform(5, 40)
        humidity = np.random.uniform(10, 90)
        oxygen = np.random.uniform(15, 21)

        if temp > 25 and humidity < 40 and oxygen < 19:
            fire = 1
        else:
            fire = 0

        if np.random.rand() < 0.05:
            fire = 1 - fire

        FFMC = np.random.uniform(80, 96)
        DMC = np.random.uniform(1, 300)
        DC = np.random.uniform(1, 800)
        ISI = np.random.uniform(0, 20)
        wind = np.random.uniform(0.5, 9)
        rain = np.random.uniform(0, 5)

        data.append([FFMC, DMC, DC, ISI, temp, humidity, wind, rain, oxygen, fire])

    columns = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'oxygen', 'fire']
    return pd.DataFrame(data, columns=columns)

def train_model():
    real_df = load_real_data()
    synth_df = generate_synthetic_data()

    all_df = pd.concat([real_df, synth_df], ignore_index=True)
    features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'oxygen']
    X = all_df[features]
    y = all_df['fire']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    lgb_clf = lgb.LGBMClassifier()
    cb_clf = cb.CatBoostClassifier(verbose=0)

    stack_model = StackingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_clf),
            ('lgb', lgb_clf),
            ('cb', cb_clf)
        ],
        final_estimator=LogisticRegression()
    )

    model = CalibratedClassifierCV(estimator=stack_model, cv=5, method='sigmoid')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    print(f"Test Accuracy: {acc}")
    print(f"Brier Score: {brier}")
    print(classification_report(y_test, y_pred))

    feature_means = dict(X.mean())
    feature_order = features
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_means': feature_means,
            'feature_order': feature_order
        }, f)
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()
