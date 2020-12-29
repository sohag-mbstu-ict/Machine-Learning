import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])
#visualizing the DecisionTree regression result
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Linear regression')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')

#visualizing the DecisionTree regression result(for high resulation and smooth)
x_grid=np.arange(min(x),max(x),.01)
x_grid=x_grid.reshape(len(x_grid),1)#1 is number of column
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('DecisionTree regression')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')