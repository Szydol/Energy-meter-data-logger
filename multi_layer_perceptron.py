import sqlite3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from math import sqrt


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix', savefile='1'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True,
                    annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidywana etykieta')
    plt.savefig(savefile)

# konfiguracja
pd.set_option('display.max_columns', None)
cnx = sqlite3.connect('C:\\Users\Szydo\\Desktop\\magisterka materiały\\do wrzucenia\\bazarefactored.db')
measurement = pd.read_sql_query("SELECT * FROM measurement_refactored", cnx)

measurement['date'] = pd.to_datetime(measurement["date"], dayfirst=True)

measurement.insert(len(measurement.columns), 'day_of_week', measurement['date'].dt.day_of_week)
measurement.insert(len(measurement.columns), 'is_weekend', (measurement['date'].dt.weekday >= 5))
measurement = measurement.replace(True, 1)
measurement = measurement.replace(False, 0)
measurement.insert(len(measurement.columns), 'hour', measurement['date'].dt.hour)
measurement.insert(len(measurement.columns), 'month', measurement['date'].dt.month)

measurement = measurement[['current', 'active_power', 'apparent_power', 'reactive_power',
                           'power_factor_value', 'presence', 'day_of_week', 'is_weekend', 'hour', 'month']]

y_data = measurement['presence']
x_data = measurement.drop('presence', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, test_size=0.2, random_state=7)


mlp_clf = MLPClassifier(random_state=10, verbose=True)
#Accuracy: 0.940201332198127
#Precision: 0.9623693788892803
#Recall: 0.9673669302490263
#F1: 0.964861683343143
mlp_clf = MLPClassifier(random_state=10, activation='tanh', solver='adam', learning_rate='invscaling', verbose=True, max_iter=200, hidden_layer_sizes=(100, 100, 100))
#Accuracy: 0.9644914108278659
#Precision: 0.9716493348050892
#Recall: 0.9869585742948188
#F1: 0.9792441230715185

mlp_clf.fit(X_train, y_train)

mlp_clf.predict(X_test)
y_pred = mlp_clf.predict(X_test)

pred_proba = mlp_clf.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1).reshape(-1, 1)

plot_confusion_matrix(cm_norm, classes=mlp_clf.classes_, title='Perceptron wielowarstwowy - obecność użytkownika', savefile='mlp_presence.png')
plt.show()
cm.sum(axis=1)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
print("The True Positive Rate is:", TPR)

# Precision or positive predictive value
PPV = TP / (TP + FP)
print("The Precision is:", PPV)

# False positive rate or False alarm rate
FPR = FP / (FP + TN)
print("The False positive rate is:", FPR)

# False negative rate or Miss Rate
FNR = FN / (FN + TP)
print("The False Negative Rate is: ", FNR)

# Total averages :
print("")
print("The average TPR is:", TPR.sum()/2)
print("The average Precision is:", PPV.sum()/2)
print("The average False positive rate is:", FPR.sum()/2)
print("The average False Negative Rate is:", FNR.sum()/2)

# errors :
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
