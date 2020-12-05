
# #### Prediction of Liver Disease
#  1. Here we are going to use Indian Liver Disease Patients datasets from Kaggle
#  2. We are going to predict whether patients has liver disease or not based on features given in dataset.


##Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

##loading dataset
data = pd.read_csv("indian_liver_patient.csv")

print(data.head())
print(data.shape)

print(data.info())

print(data.describe())

##Let's check missing values in dataset
print(data.isnull().sum())

##Albumin_and_Globulin_Ratio has 4 missing values in dataset.
data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean(), inplace = True)

#now check missing values in dataset
print(data.isnull().sum())

#### We can see that there is no missing values in dataset.

print(data.columns)

data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

#data.head()
print(data.dtypes)

print(data.columns)

##WE can see that we have balanced dataset
## 1 means patient with liver disease and 2 means patient with no-liver disease
data['Dataset'].value_counts(normalize = True)

data['Dataset'] = data['Dataset'].map({1: 1, 2: 0})

##now 1 means patient with liver disease an 0 means with no-liver disease
##Here dataset is target variable
data['Dataset'].value_counts()

##Now we split dataset into independent variable & dependent variable
X= data.iloc[:, :-1]
y = data.iloc[:, -1]

print(X.head())

print(y.head())

##Now we splitting dataset into training data & testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1234)


##Now we import RandomForest classifier algorithm 
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators= 40)

##Now we train the model
classifier.fit(x_train, y_train)

##now we predict the model on test data
predict = classifier.predict(x_test)

#predict


##let's check the performance of the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, predict)

print("Accuracy of model: ", accuracy)

cm = confusion_matrix(y_test, predict)
print(cm)

print(classification_report(y_test, predict))

##Let's check prediction on new data

new_data = pd.DataFrame([{'Age'   :55.0,
'Gender'                          :1.0,
'Total_Bilirubin'                 :1.0,
'Direct_Bilirubin'                :0.9,
'Alkaline_Phosphotase'          :277.0,
'Alamine_Aminotransferase'       :16.0,
'Aspartate_Aminotransferase'     :14.0,
'Total_Protiens'                  :4.8,
'Albumin'                         :1.3,
'Albumin_and_Globulin_Ratio'      :0.3}])


classifier.predict(new_data)

import joblib
joblib.dump(classifier,'liver_model.pkl')
