# Loading all the Libraries and packages
import os
os.chdir('C:\\Users\\shardul\\Desktop\\Rashmi\\Decision_Tree\\Heart_Disease_Decision_Tree')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


#Loading heart disease dataset

hd = pd.read_csv('HeartDisease.csv')
hd.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   age                  303 non-null    int64  
 1   gender               303 non-null    int64  
 2   chest_pain           303 non-null    int64  
 3   rest_bps             303 non-null    int64  
 4   cholestrol           303 non-null    int64  
 5   fasting_blood_sugar  303 non-null    int64  
 6   rest_ecg             303 non-null    int64  
 7   thalach              303 non-null    int64  
 8   exer_angina          303 non-null    int64  
 9   old_peak             303 non-null    float64
 10  slope                303 non-null    int64  
 11  ca                   303 non-null    int64  
 12  thalassemia          303 non-null    int64  
 13  target               303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
'''
hd.shape #303 observations & 14 variables

# Variable 1 - Target variable - target
hd.target.describe()
hd.target.value_counts()
'''
Patient has heart disease
1    165 - yes 
0    138 - no '''
hd.target.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='target', data=hd)
plt.xlabel('Patient has heart disease')
plt.ylabel('counts')
plt.title('Histogram of Patient has heart disease') 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['target'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers




## Variable 2 - Age
hd.age.describe()
hd.age.value_counts() #across different ages
hd.age.value_counts().sum() #303
hd.age.isnull().sum() #No missing values

#Histogram
plt.hist(hd.age, bins='auto')
plt.xlabel("Patient's Age")
plt.ylabel('counts')
plt.title("Histogram of Patient's age") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['age'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers




## Variable 3 -  Gender
hd.gender.describe()
hd.gender.value_counts()
'''
1    207 - Female
0     96 - male'''
hd.gender.value_counts().sum() #303
hd.gender.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='gender', data=hd)
plt.xlabel("Patient's gender")
plt.ylabel('counts')
plt.title("Histogram of Patient's gender") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['gender'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers



## Variable 4 -  Chest_pain
hd.chest_pain.describe()
hd.chest_pain.value_counts()
'''
It refers to the chest pain experienced by the patient -(0,1,2,3)
0    143
2     87
1     50
3     23'''
hd.chest_pain.value_counts().sum() #303
hd.chest_pain.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='chest_pain', data=hd)
plt.xlabel("Patient's chest_pain experience")
plt.ylabel('counts')
plt.title("Histogram of Patient's chest_pain experience") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['chest_pain'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers




## Variable 5 - rest_bps - Blood pressure of the patient while resting (in mm/Hg)
hd.rest_bps.describe()
hd.rest_bps.value_counts() #across various 
hd.rest_bps.value_counts().sum() #303
hd.rest_bps.isnull().sum() #No missing values

#Histogram
plt.hist(hd.rest_bps, bins='auto')
plt.xlabel("Blood pressure of the patient while resting")
plt.ylabel('counts')
plt.title("Histogram of Blood pressure of the patient while resting") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['rest_bps'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

iqr = hd.rest_bps.describe()['75%'] - hd.rest_bps.describe()['25%']
up_lim = hd.rest_bps.describe()['75%']+1.5*iqr
len(hd.rest_bps[hd.rest_bps > up_lim]) #9 Outliers - Ignoring outliers



## Variable 6 - cholestrol - Patient's cholesterol level (in mg/dl)
hd.cholestrol.describe()
hd.cholestrol.value_counts() #across various 
hd.cholestrol.value_counts().sum() #303
hd.cholestrol.isnull().sum() #No missing values

#Histogram
plt.hist(hd.cholestrol, bins='auto')
plt.xlabel("Patient's cholesterol level")
plt.ylabel('counts')
plt.title("Histogram of Patient's cholesterol level") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['cholestrol'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

iqr = hd.cholestrol.describe()['75%'] - hd.cholestrol.describe()['25%']
up_lim = hd.cholestrol.describe()['75%']+1.5*iqr
len(hd.cholestrol[hd.cholestrol > up_lim]) #5 Outliers - Keeping outliers

#Removing extreme outlier ie 564
hd = hd[hd.cholestrol < 500]
hd.cholestrol.describe()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['cholestrol'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers



## Variable 7 - fasting_blood_sugar 
hd.fasting_blood_sugar.describe()
hd.fasting_blood_sugar.value_counts() 
'''
0    257
1     45'''
hd.fasting_blood_sugar.value_counts().sum() #302
hd.fasting_blood_sugar.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='fasting_blood_sugar', data=hd)
plt.xlabel(" Patient's fasting_blood_sugar ")
plt.ylabel('counts')
plt.title("Countplot of Patient's fasting_blood_sugar") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['fasting_blood_sugar'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers




##  Variable 8 - rest_ecg - Potassium level (0,1,2)
hd.rest_ecg.describe()
hd.rest_ecg.value_counts() 
'''
1    152
0    146
2      4'''
hd.rest_ecg.value_counts().sum() #302
hd.rest_ecg.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='rest_ecg', data=hd)
plt.xlabel(" Patient's Potassium level ")
plt.ylabel('counts')
plt.title("Countplot of Patient's Potassium level") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['rest_ecg'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers




## Variable 9 - thalach - The patient’s maximum heart rate
hd.thalach.describe()
hd.thalach.value_counts()  #Across various
hd.thalach.value_counts().sum() #302
hd.thalach.isnull().sum() #No missing values

#Histogram
plt.hist(hd.thalach, bins='auto')
plt.xlabel(" Patient’s maximum heart rate")
plt.ylabel('counts')
plt.title("Histogram of Patient’s maximum heart rate") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['thalach'].plot.box(color=props2, patch_artist = True, vert = False) # outliers

iqr = hd.thalach.describe()['75%'] - hd.thalach.describe()['25%']
low_lim = hd.thalach.describe()['25%']-1.5*iqr
len(hd.thalach[hd.thalach < low_lim]) #1 Outlier ie 71 - Keeping outliers





## Variable 10 - exer_angina - It refers to exercise-induced angina - (1=Yes, 0=No)
#Chest discomfort or shortness of breath caused when heart muscles receive insufficient oxygen-rich blood
hd.exer_angina.describe()
hd.exer_angina.value_counts()  
'''
0    203
1     99'''
hd.exer_angina.value_counts().sum() #302
hd.exer_angina.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='exer_angina', data=hd)
plt.xlabel(" Patient exercise-induced angina")
plt.ylabel('counts')
plt.title("Countplot of Patient exercise-induced angina") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['exer_angina'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers




## Variable 11 - old_peak
'''It is the ST depression induced by exercise relative to rest(ST
relates to the position on ECG plots)'''
hd.old_peak.describe()
hd.old_peak.value_counts()  #Across various 
hd.old_peak.value_counts().sum() #302
hd.old_peak.isnull().sum() #No missing values

#Histogram
plt.hist(hd.cholestrol, bins='auto')
plt.xlabel(" Patient's old_peak")
plt.ylabel('counts')
plt.title("Histogram of Patient's old_peak") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['old_peak'].plot.box(color=props2, patch_artist = True, vert = False) #Outliers
len(hd.old_peak[hd.old_peak > 5]) #2 outliers more than 5
#Removed 2 extreme outliers
hd= hd[hd['old_peak']<5]
hd['old_peak'].plot.box(color=props2, patch_artist = True, vert = False) #Outliers
hd.info()



## Variable 12 - slope 
'''It refers to the slope of the peak of the exercise ST-Segment-(0,1,2)'''
hd.slope.describe()
hd.slope.value_counts()  
'''
2    142
1    139
0     19'''
hd.slope.value_counts().sum() #300
hd.slope.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='slope', data=hd)
plt.xlabel(" Patient's slope")
plt.ylabel('counts')
plt.title("Countplot of Patient's slope") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['slope'].plot.box(color=props2, patch_artist = True, vert = False) #No Outliers




## Variable 13 - ca - Number of major vessels - (0,1,2,3,4)
hd.ca.describe()
hd.ca.value_counts()  
'''
0    173
1     65
2     38
3     19
4      5'''
hd.ca.value_counts().sum() #300
hd.ca.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='ca', data=hd)
plt.xlabel(" Patient's ca")
plt.ylabel('counts')
plt.title("Countplot of Patient's ca") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['ca'].plot.box(color=props2, patch_artist = True, vert = False) #24 Outliers





## Variable 14 - thalassemia - It refers to thalassemia which is a blood disorder - (0,1,2,3)
hd.thalassemia.describe()
hd.thalassemia.value_counts()  
'''
2    166
3    114
1     18
0      2'''
hd.thalassemia.value_counts().sum() #300
hd.thalassemia.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='thalassemia', data=hd)
plt.xlabel(" Patient's thalassemia")
plt.ylabel('counts')
plt.title("Countplot of Patient's thalassemia") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['thalassemia'].plot.box(color=props2, patch_artist = True, vert = False) #2 outliers



#Finding the correlation
hd.corr().target.sort_values()
'''
Out[267]: 
exer_angina           -0.436500
old_peak              -0.430132
ca                    -0.390036
thalassemia           -0.343262
gender                -0.283129
age                   -0.228549
rest_bps              -0.136045
cholestrol            -0.120456
fasting_blood_sugar   -0.030004
rest_ecg               0.141279
slope                  0.339263
thalach                0.417896
chest_pain             0.428228
target                 1.000000
Name: target, dtype: float64
'''
sns.heatmap(hd.corr())

hd1=hd
hd1.shape #300, 14

#Predictors
x = hd1.iloc[:,:13]

#Respond / target variable
y = hd1.iloc[:,13]

#Partitioning the data into train/test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=123)

len(x_train) #210
len(x_test) #90


#Building Tree
clf = tree.DecisionTreeClassifier()
hd_clf = clf.fit(x_train, y_train)

#Plotting Tree
fig, ax = plt.subplots(figsize=(20, 20))
tree.plot_tree(hd_clf, ax=ax, fontsize=8,filled=True)
plt.show()

#Prediction on test data
y_pred = hd_clf.predict(x_test)
len(y_pred)
print(y_pred)

#Confusion Matrix & Report
pd.crosstab(y_test,y_pred, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        28  15   43
1        15  32   47
All      43  47   90'''

#Accuracy Score
print(accuracy_score(y_test,y_pred)) #0.66

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)
print(fpr,tpr )
#[0.         0.39534884 1.        ] [0.         0.70212766 1.        ]

#AUC -Area Under Curve
roc_auc = auc(fpr,tpr)
print(roc_auc) #0.66

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report

print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.65      0.65      0.65        43
           1       0.68      0.68      0.68        47

    accuracy                           0.67        90
   macro avg       0.67      0.67      0.67        90
weighted avg       0.67      0.67      0.67        90'''




############ BAGGING FOR 300 TREES #####################

#Base estimator is clf
#Build Bagging Classifier bc
hd_bc = BaggingClassifier(base_estimator=clf, n_estimators=300, oob_score=True, n_jobs=-1)

#Bagging Classifier fitting with training data set
hd_bc.fit(x_train, y_train)    

#Predictions
y_predbc = hd_bc.predict(x_test)
print(y_predbc)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predbc, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        32  11   43
1        10  37   47
All      42  48   90'''

#Accuracy Score
print(accuracy_score(y_test,y_predbc)) #0.77

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_predbc)
print(fpr, tpr)
#[0.         0.25581395 1.        ] [0.         0.78723404 1.        ]

#AUC -Area Under Curve
bc_roc_auc = auc(fpr,tpr)
print(bc_roc_auc) #0.77

#ROC Curve
plt.title('ROC Curve for Heart Disease with bagging')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(bc_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report

print(classification_report(y_test, y_predbc))
'''
              precision    recall  f1-score   support

           0       0.76      0.74      0.75        43
           1       0.77      0.79      0.78        47

    accuracy                           0.77        90
   macro avg       0.77      0.77      0.77        90
weighted avg       0.77      0.77      0.77        90'''




########### Random Forest ############################


# Create Model with 500 trees
rf = RandomForestClassifier(n_estimators=500,bootstrap=True,max_features='sqrt')

#Fitting the model
hd_rf = rf.fit(x_train, y_train)

#Prediction
y_predrf = hd_rf.predict(x_test)
len(y_predrf)
print(y_predrf)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predrf, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Out[44]: 
Predict   0   1  All
Actual              
0        35   8   43
1        10  37   47
All      45  45   90'''

#Accuracy Score
print(accuracy_score(y_test,y_predrf)) #0.80

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_predrf)
print(fpr,tpr)
#[0.         0.20930233 1.        ] [0.         0.80851064 1.        ]


#AUC -Area Under Curve
rf_roc_auc = auc(fpr,tpr)
print(rf_roc_auc) #0.80

#ROC Curve
plt.title('ROC Curve for Heart Disease - Random Forest ')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(rf_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
print(classification_report(y_test, y_predrf))
'''
              precision    recall  f1-score   support

           0       0.78      0.81      0.80        43
           1       0.82      0.79      0.80        47

    accuracy                           0.80        90
   macro avg       0.80      0.80      0.80        90
weighted avg       0.80      0.80      0.80        90'''

#Probabilities
yp_predrf = hd_rf.predict_proba(x_test)[:,1]
print(yp_predrf)
len(yp_predrf)


#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, yp_predrf)
print(fpr,tpr)

#AUC
rfp_roc_auc = auc(fpr,tpr)
print(rfp_roc_auc) #.89

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(rfp_roc_auc))
plt.legend(loc=4)
plt.show()

#Importance of variables
#Extract Feature importance
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': hd_rf.feature_importances_}).\
                    sort_values('importance', ascending=False)

#Display
fi.head()
'''
        feature  importance
12  thalassemia    0.141911
2    chest_pain    0.121503
11           ca    0.114998
7       thalach    0.104726
0           age    0.087574'''

####################### ADAPTIVE BOOSTING #######################

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
adafit = ada.fit(x_train, y_train)


#Prediction
y_predada = adafit.predict(x_test)
len(y_predada)
print(y_predada)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predada, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        33  10   43
1        14  33   47
All      47  43   90'''

#Accuracy Score
print(accuracy_score(y_test,y_predada)) #0.73

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_predada)
print(fpr,tpr)
#[0.         0.23255814 1.        ] [0.         0.70212766 1.        ]


#AUC -Area Under Curve
ada_roc_auc = auc(fpr,tpr)
print(ada_roc_auc) #0.73

#ROC Curve
plt.title('ROC Curve for Heart Disease - Adaptive Boosting')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(ada_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
print(classification_report(y_test, y_predada))
'''
              precision    recall  f1-score   support

           0       0.70      0.77      0.73        43
           1       0.77      0.70      0.73        47

    accuracy                           0.73        90
   macro avg       0.73      0.73      0.73        90
weighted avg       0.74      0.73      0.73        90'''

#Probabilities
yp_predada = adafit.predict_proba(x_test)[:,1]
print(yp_predada)
len(yp_predada)

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, yp_predada)

#AUC -Area Under Curve
adap_roc_auc = auc(fpr,tpr)
print(adap_roc_auc) #0.84

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(adap_roc_auc))
plt.legend(loc=4)
plt.show()

