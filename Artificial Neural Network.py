
#link : https://www.superdatascience.com/blogs/the-ultimate-guide-to-artificial-neural-networks-ann
#link : https://towardsdatascience.com/introduction-to-artificial-neural-networks-ann-1aea15775ef9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
xx= dataset.iloc[:,3:13]
y = dataset.iloc[:, 13].values
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1=LabelEncoder()
x[:,1]=labelencoder_x1.fit_transform(x[:,1])
labelencoder_x2=LabelEncoder()
x[:,2]=labelencoder_x2.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer

x[: , 1] = labelencoder_x1.fit_transform(x[:,1])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
x = transformer.fit_transform(x.tolist())
x = x.astype('float64')
x=x[:,1:]#first column bade baki sob

#onehotencoder=OneHotEncoder(categorical_features=[1])
#x=onehotencoder.fit_transform(x).toarray()
#x=x[:,1:]#first column bade baki sob
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#import tensorflow libraries and package
import tensorflow as tf
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#initializing the ANN (object)
classifier=Sequential()
#adding the input layer and the first hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
#unit holo hidden layer=(input+output)/2
#input_dim=number of independent variable

#adding the second hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
#adding the hidden layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ANN to the training set
classifier.fit(x_train,y_train,batch_size=10,epochs=100)
#predict the test set result
y_pred=classifier.predict(x_test)
y_pred1=(y_pred>0.5)#if y_pred>0.5 then return True(customer leave the bank)
                    #else return False(customer does not leave the bank)

#evaluate accuracy using confusion_metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred1)



