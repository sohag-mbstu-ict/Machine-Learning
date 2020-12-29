import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='mean',axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
x[:,1:3]

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:, 0])
#now we use dummy variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])#0 number column
x=onehotencoder.fit_transform(x).toarray()
x
#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)
y_train
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(x_train)
sc.transform(x_test)  #here we does not need fit because x_test get automatically fit
#####we dont need to scalling the y that is target variable