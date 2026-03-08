
import time
from sklearn.metrics import precision_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBClassifier

scaler = StandardScaler()
df = pd.read_csv('creditcard.csv')
X = df.iloc[:,:-1].dropna()
y = df.iloc[:,-1].dropna()

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start = time.time()
    xgb = XGBClassifier(random_state=0, n_estimators=50, max_depth=7, scale_pos_weight = 10, gamma=1)
    xgb.fit(X_train_scaled, y_train)
    end = time.time()

    y_pred = xgb.predict(X_test_scaled)
    y_prob = xgb.predict_proba(X_test_scaled)[:,1]

    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    fhand = open('XGData.txt', 'a')
    fhand.write('''
    \nXGBoost {}
    Precision score: {}
    Recall score: {}        
    roc_auc score: {}
    training time: {} second'''.format(i+1, pre, rec, roc, end-start))