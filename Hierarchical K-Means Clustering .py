import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
#using the dendrogram method to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
#if method='ward' then it is a user responsibility to assure that these distances 
#are in fact Euclidean
##if method='ward' minimize the variance each of the cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian distance')
#fitting hierarchical clustering to the Mall_Customers dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
#affinity is a distance method
y_hc=hc.fit_predict(x)
#visualizing the hierarchical clustering
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=80,c='red',label='careful')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=80,c='blue',label='standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=80,c='green',label='target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=80,c='cyan',label='careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=80,c='magenta',label='sensible')
plt.title('clusters of client')
plt.xlabel('annual income K$')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()