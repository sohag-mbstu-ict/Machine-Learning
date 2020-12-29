import pandas as pd
import numpy as np

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,0:1].values
x
y=dataset.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#visualize the training set result
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experiences (training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')

#visualize the test set result
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experiences (test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')