# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: K SARANYA
RegisterNumber: 212224040298 
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

## data.head()

<img width="339" height="258" alt="image" src="https://github.com/user-attachments/assets/54ca542a-5036-4eeb-b67f-5122d60fcb09" />

## data.info()

<img width="428" height="230" alt="image" src="https://github.com/user-attachments/assets/e7219064-9996-4620-8684-70c165783429" />

## data.isnull().sum()

<img width="209" height="109" alt="image" src="https://github.com/user-attachments/assets/80a7c035-8620-4883-be29-3a16d98b5edc" />

## data.head() for salary

<img width="280" height="253" alt="image" src="https://github.com/user-attachments/assets/2087f7f4-f0c5-4639-9b3b-e6061fef658d" />

## MSE value

<img width="265" height="54" alt="image" src="https://github.com/user-attachments/assets/7470cdab-50fc-4913-a0db-d34807f8922d" />

## r2 value

<img width="329" height="55" alt="image" src="https://github.com/user-attachments/assets/29fe28eb-ff68-4d64-97bf-6eb4e9fd23fb" />

## data prediction

<img width="1664" height="98" alt="image" src="https://github.com/user-attachments/assets/93d90af4-f67a-4144-9a54-70b9522f10db" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
