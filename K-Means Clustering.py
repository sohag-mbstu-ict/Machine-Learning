#importing dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importng the Mall dataset using pandas
dataset=pd.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,[3,4]].values
#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[] #Within-Cluster-Sum-of-Squares (WCSS)
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    #n_init : j koi bar initialize hobe and valo value kujbe
    #init='k-means++' : j point theke suru hobe ta nirdaron korbe
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)#kmeans.inertia_ does claculate Cluster sum of square of point
    
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel('the number ranges')
plt.ylabel('wcss')
plt.show()
#applying k-means to the Mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(x)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=80,c='red',label='careful')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=80,c='blue',label='standard')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=80,c='green',label='target')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=80,c='cyan',label='careless')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=80,c='magenta',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=270,c='yellow',label='centroides')
plt.title('clusters of client')
plt.xlabel('annual income K$')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()
