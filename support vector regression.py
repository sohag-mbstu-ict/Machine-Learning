import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)
#it gives us scalling prediction output
y_pred=regressor.predict(sc_x.transform(np.array([[6.5]])))
#it gives us orginal prediction output
y_pred1=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('support vector regression')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
