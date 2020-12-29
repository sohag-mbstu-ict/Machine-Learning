import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
#Implementing Thompson Sampling Algorithm
import random
N=10000
d=10
ads_selected=[]
numbers_of_reward_1=[0]*d
numbers_of_reward_0=[0]*d
sums_of_reward=0
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(numbers_of_reward_1[i]+1,numbers_of_reward_0[i]+1)
        if(random_beta>max_random):
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if(reward==1):
        numbers_of_reward_1[ad]=numbers_of_reward_1[ad]+1
    else:#update na korle sudu 0 pabe ,,,karon 10 ta 0 dara initialize korsi 
        numbers_of_reward_0[ad]=numbers_of_reward_0[ad]+1
    sums_of_reward=sums_of_reward+reward
 #visualizing the results
 plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
 