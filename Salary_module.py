#this code is to import the numpy library
import numpy as np

#this code is to import pandas
import pandas as pd

#this code is to import the matplotlib library
import matplotlib.pyplot as plt

#this code is to create a variable to store dataset
Salary_Data = pd.read_csv("Salary_Data.csv")

#create variable x to store the independent column values
x = Salary_Data.iloc[:,0:1].values
y = Salary_Data.iloc[:,1:2].values

#splitting the dataset into train data and test data
from sklearn.model_selection import train_test_split

#create variable to store x_train, x_test and y_train, y_test 
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=1/3,random_state=0) 

#training the linear regression module
from sklearn.linear_model import LinearRegression 

#create a variable and assigning Linear regression 
DE_Module = LinearRegression()

#training the module SD with x_train and y_train
DE_Module.fit(x_train,y_train)

#making a prediction
prediction_result = DE_Module.predict(x_test)

prediction_result
