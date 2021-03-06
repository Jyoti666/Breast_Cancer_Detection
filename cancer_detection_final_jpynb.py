# -*- coding: utf-8 -*-
"""Cancer_Detection_Final.jpynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iXqRMeG0yeItr5a8C1tChPibLTzVWB9H
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

data=load_breast_cancer()

data.feature_names

data

data_frame=pd.DataFrame(np.c_[data['data'],data['target']],columns=np.append(data['feature_names'],'target'))
data_frame.head()

len(data.feature_names)

data_frame.shape

data_frame.isna().sum()

#Manage the Missing Values
#data_frame.dropna(axis=1)

#Count No. of Benign and Malignant
data_frame['target'].value_counts()

#Visualize No. of Benign and Malignant
sns.countplot(data_frame['target'],label='count')

#Data type of attributes
data_frame.dtypes

# create a pair plot
sns.pairplot(data_frame.iloc[:,0:30])

#get corelation of the columns
data_frame.iloc[:,0:30].corr()

#visualize correlation
plt.figure(figsize=(10,10))
sns.heatmap(data_frame.iloc[:,0:11].corr(),annot=True, fmt='.0%')

x=data_frame.iloc[:,:30]
x.head()

y=data_frame['target']
y.head()

#Deal with Missing values
"""imputer=SimpleImputer(missing_values=np.NaN,strategy='mean')
imputer=imputer.fit(x.iloc[:,:30])
x.iloc[:,0:30]=imputer.transform(x.iloc[:,:30])
x"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
x_train.head()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.transform(x_test)



#Without Standard Scalling
from sklearn.svm import SVC
svc_classifier=SVC()
svc_classifier.fit(x_train,y_train)
svc_predict=svc_classifier.predict(x_test)
accuracy_score(y_test,svc_predict)*100

#With Standard Scalling
from sklearn.svm import SVC
svc_classifier_sc=SVC()
svc_classifier_sc.fit(x_train_sc,y_train)
svc_predict_sc=svc_classifier_sc.predict(x_test_sc)
accuracy_score(y_test,svc_predict_sc)*100



#Without Standard Scalling
from sklearn.linear_model import LogisticRegression
lr_classifier=LogisticRegression()
lr_classifier.fit(x_train,y_train)
lr_predict=lr_classifier.predict(x_test)
accuracy_score(y_test,lr_predict)*100

#With Standard Scalling
from sklearn.linear_model import LogisticRegression
lr_classifier_sc=LogisticRegression()
lr_classifier_sc.fit(x_train_sc,y_train)
lr_predict_sc=lr_classifier_sc.predict(x_test_sc)
accuracy_score(y_test,lr_predict_sc)*100



#Without Standard Scalling
from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn_classifier.fit(x_train,y_train)
knn_classifier_predict=knn_classifier.predict(x_test)
accuracy_score(y_test,knn_classifier_predict)*100

#With Standard Scalling
from sklearn.neighbors import KNeighborsClassifier
knn_classifier_sc=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn_classifier_sc.fit(x_train_sc,y_train)
knn_classifier_predict_sc=knn_classifier_sc.predict(x_test_sc)
accuracy_score(y_test,knn_classifier_predict_sc)*100



#Without Standard Scalling
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
nb_predict=nb.predict(x_test)
accuracy_score(y_test,nb_predict)*100

#With Standard Scalling
from sklearn.naive_bayes import GaussianNB
nb_sc=GaussianNB()
nb_sc.fit(x_train_sc,y_train)
nb_predict_sc=nb_sc.predict(x_test_sc)
accuracy_score(y_test,nb_predict_sc)*100



#Without Standard Scalling
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy',random_state=51)
dt.fit(x_train,y_train)
dt_predict=dt.predict(x_test)
accuracy_score(y_test,dt_predict)*100

#With Standard Scalling
from sklearn.tree import DecisionTreeClassifier
dt_sc=DecisionTreeClassifier(criterion='entropy',random_state=51)
dt_sc.fit(x_train_sc,y_train)
dt_predict_sc=dt_sc.predict(x_test_sc)
accuracy_score(y_test,dt_predict_sc)*100



#Without Standard Scalling
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=51)
rf.fit(x_train,y_train)
rf_predict=rf.predict(x_test)
accuracy_score(y_test,rf_predict)*100

#With Standard Scalling
from sklearn.ensemble import RandomForestClassifier
rf2=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=51)
rf2.fit(x_train_sc,y_train)
rf_predict2=rf2.predict(x_test_sc)
accuracy_score(y_test,rf_predict2)*100



#Without Standard Scalling
from sklearn.ensemble import AdaBoostClassifier
ab=AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1)
ab.fit(x_train,y_train)
ab_predict=ab.predict(x_test)
accuracy_score(y_test,ab_predict)*100

#With Standard Scalling
from sklearn.ensemble import AdaBoostClassifier
ab2=AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1)
ab2.fit(x_train_sc,y_train)
ab_predict2=ab2.predict(x_test_sc)
accuracy_score(y_test,ab_predict2)*100



#Without Standard Scalling
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
xgb_predict=xgb.predict(x_test)
accuracy_score(y_test,xgb_predict)*100

#With Standard Scalling
from xgboost import XGBClassifier
xgb2=XGBClassifier()
xgb2.fit(x_train_sc,y_train)
xgb_predict2=xgb2.predict(x_test_sc)
accuracy_score(y_test,xgb_predict2)*100

#Without Standard Scalling
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=10000)
mlp.fit(x_train,y_train)
y_predict=mlp.predict(x_test)
accuracy_score(y_test,y_predict)*100

#With Standard Scalling
mlp2=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=10000)
mlp2.fit(x_train_sc,y_train)
y_predict2=mlp2.predict(x_test_sc)
accuracy_score(y_test,y_predict2)*100

# XGBoost classifier most required parameters
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}

# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(xgb, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3       ) 
random_search.fit(x_train, y_train)

random_search.best_params_

random_search.best_estimator_

xgb_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
       learning_rate=0.1, max_delta_step=0, max_depth=15,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
 
xgb_pt.fit(x_train, y_train)
xgb_pt_predict = xgb_pt.predict(x_test)

accuracy_score(y_test,xgb_pt_predict)*100



cm=confusion_matrix(y_test,xgb_pt_predict)
plt.title('Heatmap of Confusion Matrix',fontsize=15)
sns.heatmap(cm,annot=True)
plt.show()

print(classification_report(y_test,xgb_pt_predict))

# Cross validation

from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_pt, X = x_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())





import pickle
pickle.dump(xgb_pt,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
predict=model.predict(x_test)
accuracy_score(y_test,predict)*100