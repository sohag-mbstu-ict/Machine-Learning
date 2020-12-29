#import libraries
import numpy as np
import os
import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense
df=pd.read_csv('bankloan.csv')
dff=df
#https://stackoverflow.com/questions/36226083/how-to-find-which-columns-contain-any-nan-value-in-pandas-dataframe
column_null=df.isna().any()
#if column hava Nan or null value then return True, null value na thakle return False
column_null_list=df.columns[df.isna().any()].tolist()
#j j column a Nan or null value ase tar list show korbe
column_null_location=df.loc[:, df.isna().any()]
df=df.dropna()#remove Nan or null values row
df=df.drop('Loan_ID',axis=1)
df['LoanAmount']=(df['LoanAmount']*1000).astype(int)
loan_status=Counter(df['Loan_Status'])#koto gulo yes or no ase 
probability_of_yes=Counter(df['Loan_Status'])['Y']/df['Loan_Status'].size######

pre_y=df['Loan_Status']
pre_x=df.drop('Loan_Status',axis=1)
dm_x=pd.get_dummies(pre_x)
dm_y=pre_y.map(dict(Y=1,N=0))

smote = SMOTE(sampling_strategy='minority')
x1,y=smote.fit_sample(dm_x, dm_y)
#number of row will be increase because minority more new data will be added
#to equal to the number of majority data (using average calculation & other calculation)  
sc=MinMaxScaler()
x=sc.fit_transform(x1)
balance_loan_status=Counter(y)#1 for yes, 0 for no

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
shape_x_test=x_test.shape[1]#that means number of column###
classifier=Sequential()
#adding  hidden layer
classifier.add(Dense(200,activation='relu',kernel_initializer='random_normal',input_dim=x_test.shape[1]))
classifier.add(Dense(400,activation='relu',kernel_initializer='random_normal'))
classifier.add(Dense(4,activation='relu',kernel_initializer='random_normal'))
#adding output layer
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ANN to the training set
classifier.fit(x_train,y_train,epochs=100,batch_size=20,verbose=0)
#evaluate the  accuracy
evaluate_acuracy=classifier.evaluate(x_train,y_train)
#predict the test set result
y_pred=classifier.predict(x_test)
y_pred1=(y_pred>0.5)
approve_reject=pd.DataFrame(y_pred1,columns=['status'])####
approve_reject=approve_reject.replace({True:'Approved',False:'Rejected'})###
#cm=confusion_matrix(y_test,y_pred1)
#ax=plt.subplot()
#sns.heatmap(cm,annot=True,ax=ax)
#ax.set_xlabel('Predicted')
#ax.set_ylabel('Actual')
#ax.set_title('Confusion Matrix')
#ax.xaxis.set_ticklabels(['No','Yes'])
#ax.yaxis.set_ticklabels(['No','Yes'])
#https://tutorialspoint.dev/language/python/saving-a-machine-learning-model 
classifier.summary()
#save the model (in tensorflow rules)
#https://medium.com/next-gen-machine-learning/keras-save-model-and-keras-load-model-d516d6234776
#classifier.save('bank_loan.h5')
classifier.save('bank_loan.h5') 
#save the model (in tensorflow rules)
new_model=keras.models.load_model('bank_loan.h5')
#test the model
new_y_pred1=new_model.predict(x_test)
new_y_pred1=(new_y_pred1>0.5)
#model1=joblib.load('loan_model.pkl')
