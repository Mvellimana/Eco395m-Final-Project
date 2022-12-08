# Naive Bayes: TF IDF

"""Import Packages"""
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

"""Import the Dataset"""
yelp_data = pd.read_csv(r'/Users/rajsitee/Desktop/VS Code trial copy/Eco395m-Final-Project/Cleaned_reviews.csv')

"""Convert polarity into a binary classification. 
For all values < 0, polarity = 0. For all values >= 0, polarity = 1. """
temp_array = list(yelp_data.polarity)

for i in range (len(temp_array)):
    if temp_array[i] < 0:
        yelp_data.loc[i, 'polarity']= 0
    elif temp_array[i] >= 0:
        yelp_data.loc[i, 'polarity']= 1
        
"""Train-test split + random state"""        
X = yelp_data.cleaned_review
y = yelp_data.polarity
indices = yelp_data.index

X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(X, y, indices, train_size = 0.8, random_state = 7)

"""Model Validation"""
steps = [('vec', TfidfVectorizer(stop_words = 'english', ngram_range = (1, 2))), ('nb', MultinomialNB())] 
pipeline = Pipeline(steps) 
parameters = {'vec__min_df':[0.01, 0.1, 1, 10, 100], 'nb__alpha':[0.01, 0.1, 1, 10, 100]}

clf = GridSearchCV(pipeline, parameters, cv = 10, scoring="accuracy") 
clf.fit(X_train, y_train)

clf.best_params_

"""Save the file into an sav"""
filename = 'nb_tf.sav'
pickle.dump(clf, open(filename, 'wb'))

"""Load the model file"""
filename = 'nb_tf.sav'
clf = pickle.load(open(filename, 'rb'))

"""Results"""
results = clf.predict(X_test)

"""Test the Accuracy & the F1 Score of the Model"""
test_accuracy = clf.score(X_test, y_test)
probs = clf.predict_proba(X_test)[:, 1]
f1_accuracy = f1_score(y_test,results,average='macro')
f1_accuracym = f1_score(y_test,results,average='micro')
f1_accuracyw = f1_score(y_test,results,average='weighted')
print("Accuracy on test data: " ,test_accuracy)
print('F1 Score (macro): ', f1_accuracy)
print('F1 Score (micro): ', f1_accuracym)
print('F1 Score (weighted): ', f1_accuracyw)



