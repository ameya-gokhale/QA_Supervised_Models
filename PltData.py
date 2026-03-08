
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # type: ignore
import re

fhand = open('RFData.txt', 'r')
data = fhand.read()

X_label = ['RandomForest', 'Logistic Regression', 'XGBoost']
fnames=['RFData.txt', 'LogData.txt', 'XGData.txt']
plt.figure(figsize=(12,8))

avg_rec, avg_pre, avg_roc, avg_tme = [], [], [], []
for fname in fnames:
    fhand = open(fname)
    data = fhand.read()
    pos = fnames.index(fname)+1
    
    rec_rf = list(map(float, re.findall(r"Recall score: (\d+\.\d*)", data)))
    pre_rf = list(map(float, re.findall(r"Precision score: (\d+\.\d*)", data)))
    roc_rf = list(map(float,re.findall(r"roc_auc score: (\d+\.\d*)", data)))
    tme_rf = list(map(float,re.findall(r"training time: (\d+\.\d*)", data)))

    plt.subplot(2,2,1)
    plt.title("Recall score")
    plt.boxplot(rec_rf, positions=[pos])
    plt.xticks([1, 2, 3], X_label)
    plt.ylabel('Recall')

    plt.subplot(2,2,2)
    plt.title("Precision score")
    sns.stripplot(x=[X_label[pos-1]]*10, y=pre_rf, jitter=True)
    plt.ylabel('Precision')

    plt.subplot(2,2,3)
    plt.title("roc_auc score")
    plt.boxplot(roc_rf, positions=[pos])
    plt.xticks([1, 2, 3], X_label)
    plt.ylabel('auc score')

    plt.subplot(2,2,4)
    plt.title("Time required")
    sns.stripplot(x=[X_label[pos-1]]*10, y=tme_rf, jitter=True)
    plt.yscale('log')
    plt.ylabel('Time required')

    plt.savefig('Analysis.png', dpi=300, bbox_inches='tight')

    avg_rec.append(sum(rec_rf)/len(rec_rf))
    avg_pre.append(sum(pre_rf)/len(pre_rf))
    avg_tme.append(sum(tme_rf)/len(tme_rf))
    avg_roc.append(sum(roc_rf)/len(roc_rf))

angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

for i in range(3):
    values = [avg_pre[i], avg_rec[i], avg_roc[i]]
    ax.plot(angles, values, label=X_label[i])
    ax.fill(angles, values, alpha=0.25)

for angle, label in zip(angles, ['Precision', 'Recall', 'roc_auc']):
    ax.text(angle, 1.15, label)

plt.title('Avg scores of models', loc='center', pad=20)
ax.legend(bbox_to_anchor=(1.3, 1.1))
ax.set_xticks([]) 
plt.savefig('Avg scores.png', dpi=300, bbox_inches='tight')

