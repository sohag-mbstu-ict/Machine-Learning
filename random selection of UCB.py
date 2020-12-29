import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
import random
N=10000
d=10
total_reward=0
ads_selected=[]
for i in range(0,N):
    ad=random.randrange(d)#0-9 theke j kno ta nibo
    ads_selected.append(ad)
    reward=dataset.values[i,ad]
    total_reward=reward+total_reward

plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('ads')
plt.ylabel('number of times each ads was selected')
plt.show()