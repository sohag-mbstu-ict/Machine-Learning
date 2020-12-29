import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values # or x=dataset.drop('State',axis=1)
y=dataset.iloc[:,4]

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#avoiding the dummy variable trap
x=x[:,1:]#to avoid dummy variable trap we delete one column that is (California)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
import statsmodels.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]#copy the index 0 to 5 from x in x_opt
#OLS: ordinary least square
regressor_OLS=sm.OLS(exog=x_opt,endog=y).fit()
regressor_OLS.summary()
#first we adjust with the main x and find which column will ve removed
#here dummy variable is x0 then x1,x2,x3, and others 
x_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(exog=x_opt,endog=y).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(exog=x_opt,endog=y).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3,5]]
regressor_OLS=sm.OLS(exog=x_opt,endog=y).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3]]
regressor_OLS=sm.OLS(exog=x_opt,endog=y).fit()
regressor_OLS.summary()