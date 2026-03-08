
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('creditcard.csv')
X = df.iloc[:,:-1].dropna()
y = df.iloc[:,-1].dropna()

scaler = StandardScaler()

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start = time.time()
    model = LogisticRegression(random_state=0, C=0.001, penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    end=time.time()

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    pre = precision_score(y_test, y_pred)

    fhand = open('LogData.txt', 'a')
    fhand.write('''
    \nLogistic Regression {}
    Precision score: {}
    Recall score: {}        
    roc_auc score: {}
    training time: {} second'''.format(i+1, pre, rec, roc, end-start))