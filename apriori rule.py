#link : https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]#it's a list
for i in range(0,7501):
    transactions.append(str([dataset.values[i,j] for j in range(0,20)]))
#[dataset.values[i,j] for j in range(0,21)] ,[] is for to make list
# j is a number of column
#each row of a dataset is list that is added to the broder one list that is transations list 

from apriori import apriori
#here apriori is a free API that is included in file explorer
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
#min_confidence=0.2 means all the row will be true at least 20%
#min_lift=3 means three times true of bought b if a is bought than bought of a & b together
#min_length=2 means rows consist with minimum two(2) product

results=list(rules)