## TextBlob Naïve Bayes Classifier Sentiment Analyzer ##

"""Imports"""
import pandas as pd
import os
import nltk
nltk.download('punkt')
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split

"""Import the Dataset"""
PATH_IN = os.path.join("artifacts", "Cleaned_reviews.csv")
df = pd.read_csv(PATH_IN)

"""One-hot-encoding"""
temp_array = list(df.polarity)

for i in range (len(temp_array)):
    if temp_array[i] < 0:
        df.loc[i, "polarity"]= 0
    elif temp_array[i] >= 0:
        df.loc[i, "polarity"]= 1

"""Out of all the positive samples, randomly pick 7,852 positive samples in a new dataframe. 
Create a new dataframe to hold all the negative samples.
Concatenate the two dataframes to form a joint df."""

df_positive = df[(df["polarity"] == 1)]
df_positive_sample = df_positive.sample(n=7852)
 
df_negative = df[(df["polarity"] == 0)]
df_negative_sample = df_negative.sample(n=7852)

yelp = pd.concat([df_positive_sample, df_negative_sample], axis=0)

"""Train-Test Split"""
X = yelp.cleaned_review
y = yelp.polarity
indices = yelp.index

X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(X, y, indices, train_size = 0.8, random_state = 7)
print(X_train)

"""model validation"""
sample = pd.DataFrame([X_train, y_train]).transpose()
sample['desig'] = sample.polarity.apply(lambda x: 'pos' if x == 1 else 'neg')

input_train = []
for text, d in list(zip(sample['cleaned_review'], sample['desig'])): 
    input_train.append((text, d))
print(input_train)

sample2 = pd.DataFrame([X_test, y_test]).transpose()
sample2['desig'] = sample2.polarity.apply(lambda x: 'pos' if x == 1 else 'neg')

input_test = []
for text, d in list(zip(sample['cleaned_review'], sample['desig'])): 
    input_test.append((text, d))
    
print(input_test)

cl = NaiveBayesClassifier(input_train)

"""Results"""
cl.accuracy(input_test)



