#link : https://medium.com/pursuitnotes/day-30-31-natural-language-processing-2-7fa9d4558426
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Importing Dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#quoting=3 is used for ignoring double quotaton("")
#cleaning the text
import re
import nltk#it helps to delete unimportant words(preposition,article,conjunction etc)
nltk.download('stopwords')#Text may contain stop words like ‘the’, ‘is’, ‘are’.
#it convert each row of Review into list element
#stopwords is a package contain different list
from nltk.corpus import stopwords#corpus is collection of text
#cleaning the irrelevant word by using stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
#verb er j kno form theke present form a convert korbe
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    #re.sub('[^a-zA-Z]',') letter bad dia baki sob space dara replace korbe 
    review=review.lower()#all letter will be lower case
    review=review.split()#This converts the review string to a list of words.
    
    review=[word for word in review if not word in set(stopwords.words('english'))]
    #In the for loop above, we’re taking all the words in the review that
    # are not in this stopwords’ list and more precisely the list of 
    #English words that are not relevant in text.
    
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #stemming process: It is taking the root(verb er present form) of the word as it is declared above
    review=' '.join(review)#rejoin the words to make the line string again.
    corpus.append(review)

#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()#since it is matrix so used toarray()
#cv create sparse matrix(lots of zero contains is called sparse)
#each different word will have different column of its own
#jemon 'wow' er jonno one column a 1 hobe jodi r kno row te r 'wow' na thake tobe ta o hobe 
y=dataset.iloc[:,1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy of the model
from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,y_pred)

