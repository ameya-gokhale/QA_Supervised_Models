
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('creditcard.csv')
X = df.iloc[:,:-1].dropna()
y = df.iloc[:,-1].dropna()

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    start = time.time()
    model = RandomForestClassifier(random_state=0, max_depth=5, n_estimators=200, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    end=time.time()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc =  roc_auc_score(y_test, y_prob)

    fhand = open('RFData.txt', 'a')
    fhand.write('''
    \nRandomForestClassifier {}
    Precision score: {}
    Recall score: {}        
    roc_auc score: {}
    training time: {} second'''.format(i+1, pre, rec, roc, end-start))
