# smote over sampling for unbiased dataset and better precision.



import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sb

import xgboost as xgb
from xgboost import XGBClassifier



from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay


import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('./train.csv')
df_train.head(2)

df_train['is_anomaly'] = df_train['is_anomaly'].replace(False,0).replace(True,1)
df_train['is_anomaly'].value_counts()

df_train.isnull().sum()


df_train.describe()

plt.figure(figsize=(25, 9))
sb.heatmap(df_train.corr(),annot=True,cmap='coolwarm')
plt.show()

sb.scatterplot(x=df_train['predicted'], y=df_train['value'])

print("Total No of Transactions:",df_train.size)

Fraud_df = df_train[df_train['is_anomaly']==True]
print("No of Anomalous Transactions:",len(Fraud_df))

Valid_df = df_train[df_train['is_anomaly']==False]
print("No of Valid Transactions:",len(Valid_df))

outlier_fraction = len(Fraud_df)/float(len(df_train))
valid_fraction = len(Valid_df)/float(len(df_train))
print("Percentage of Anomalous Transactions:",round((outlier_fraction*100),3))
print("Percentage of Valid Transactions:",round((valid_fraction*100),3))

X = df_train.drop(columns=['is_anomaly'],inplace=False,axis=1)
X.head(2)

y = df_train['is_anomaly']
y.head(3)

X.shape
X_train = X.copy(deep=True)
y_train  = y.copy(deep=True)

state = np.random.RandomState(42)
X_outliers = state.uniform(low=0, high=1, size=(X_train.shape[0], X_train.shape[1]))

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100,
                                       max_samples=len(X_train), 
                                       contamination=outlier_fraction,
                                       random_state=state, 
                                       verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,
                                              algorithm='auto', 
                                              leaf_size=30, 
                                              metric='minkowski',
                                              novelty=False,
                                              p=2, metric_params=None,
                                              contamination=outlier_fraction),
    "Novelty Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                                      novelty=True,p=2, metric_params=None, 
                                                      contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1),
    "XGBClassifier":XGBClassifier(objective="binary:logistic", random_state=42)
}

f, axes = plt.subplots(1, 5, figsize=(20, 10), sharey='row')
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    print("###"*32)
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X_train)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X_train)
        y_pred = clf.predict(X_train)
    elif clf_name == "Novelty Local Outlier Factor":
        clf.fit(X_train)
        y_pred = clf.predict(X_train)
        scores_prediction = clf.negative_outlier_factor_  
    elif clf_name == "XGBClassifier":
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_train)
    else:    
        clf.fit(X_train)
        scores_prediction = clf.decision_function(X_train)
        y_pred = clf.predict(X_train)
#     Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != y_train).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    ac_score = accuracy_score(y_train,y_pred)
    
    print(f"Accuracy Score :{round(ac_score,2)}")
    print("Classification Report :")
    print(classification_report(y_train,y_pred))
    cf_matrix = confusion_matrix(y_train, y_pred)
    disp = ConfusionMatrixDisplay(cf_matrix)
    disp.plot(ax=axes[i], values_format='.0f',cmap = "Blues")
    axes[i].set_title(clf_name+"f1:"+str(round(ac_score,2)))
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')

X_test=pd.read_csv('./test.csv')

clf  = XGBClassifier(objective="binary:logistic", random_state=42)
clf.fit(X_train,y_train)
y_test_pred = clf.predict(X_test)

data={"timestamp":[],"is_anomaly":[]}
for id,pred in zip(X["timestamp"].unique(),y_test_pred):
    data["timestamp"].append(id)
    data["is_anomaly"].append(pred)

output=pd.DataFrame(data,columns=["timestamp","is_anomaly"])
output.head(2)

output['is_anomaly'].value_counts()

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
output['is_anomaly'].value_counts()