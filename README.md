# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.KIRUTHIGA
RegisterNumber: 212219040061 


import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))*/
```

## Output:
```
Original data(first five columns):
![image](https://user-images.githubusercontent.com/98682825/174330831-41afd13c-d264-4aac-a045-2eaa18b4f0f5.png)
```
```
Data after dropping unwanted columns(first five):
![image](https://user-images.githubusercontent.com/98682825/174331114-10294d7a-0e1d-4365-9d59-f2121053b689.png)
```
```
Checking the presence of null values:
![image](https://user-images.githubusercontent.com/98682825/174331182-6961ad52-1f37-4936-85f0-c9ae7a5ab626.png)
```
```
Checking the presence of duplicated values:
![image](https://user-images.githubusercontent.com/98682825/174331435-a55abf4a-e135-43ea-a9ef-38afde669f62.png)
```
Data after Encoding:
![image](https://user-images.githubusercontent.com/98682825/174331250-db11e856-ca79-4246-a83e-abc3011e2d58.png)
X Data:
![image](https://user-images.githubusercontent.com/98682825/174331300-c1c87bc7-8dc5-4d60-ad37-ca4cac8be610.png)
Y Data:
![image](https://user-images.githubusercontent.com/98682825/174331337-6bd65c3a-6050-4daa-9987-86b719f0170f.png)
Predicted Values:
![image](https://user-images.githubusercontent.com/98682825/174331535-c908b330-cb10-42f1-95bf-6c4ea70e920a.png)
Accuracy Score:
![image](https://user-images.githubusercontent.com/98682825/174331559-fe3324d7-1a00-45ac-b814-63c33640a267.png)
```
Confusion Matrix:
![image](https://user-images.githubusercontent.com/98682825/174331574-e28d99db-1860-48e2-869b-73578aee1d7f.png)
```
Classification Report:
![image](https://user-images.githubusercontent.com/98682825/174331600-c495ba26-a143-4407-8855-5ca6c915b02f.png)
Predicting output from Regression Model:
![image](https://user-images.githubusercontent.com/98682825/174331635-e869ecb9-2398-46c4-8d77-622201cf5e98.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
