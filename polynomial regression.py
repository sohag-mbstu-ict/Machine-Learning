import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
poly_x=poly_reg.fit_transform(x)
poly_reg.fit(poly_x,y)
lin_reg2=LinearRegression()
lin_reg2.fit(poly_x,y)
#visualizing the linear regression result
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear regression')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')

#visualizing the linear regression result
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear regression')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
#visualizing the polynomial regression result
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
#since prediction reqire polynomial feature x
plt.title('Linear regression')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')

#predicting a new result with linear regression
lin_reg.predict([[6.5]])

#predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))