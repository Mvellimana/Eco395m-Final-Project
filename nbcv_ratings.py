## Predictive Modelling [NLP]

# Naive Bayes: Count Vectorizer

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
yelp_data = pd.read_csv(r'/Users/rajsitee/Downloads/Eco395m-Final-Project-jordan/artifacts/mexican_reviews.csv')

"""Convert rating into a binary classification. 
For all ratings <= 3, rating_binary = 0. For all ratings >= 4, rating_binary = 1."""
def star_rating(x):

    if x.rating == '3 star rating':
        return 0
    elif x.rating == '2 star rating':
        return 0
    elif x.rating == '1 star rating':
        return 0
    elif x.rating == '4 star rating':
        return 1
    elif x.rating == '5 star rating':
        return 1
    else:
        return 0
yelp_data['rating_binary'] = yelp_data.apply(star_rating,axis = 1)

"""Train-test split + random state"""   
X = yelp_data.review
y = yelp_data.rating_binary
indices = yelp_data.index

X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(X, y, indices, train_size = 0.8, random_state = 7)
print(X_train)

"""Model Validation"""
steps = [('vec', CountVectorizer(stop_words = 'english', ngram_range = (1, 2))), ('nb', MultinomialNB())] 
pipeline = Pipeline(steps) 
parameters = {'vec__min_df':[0.01, 0.1, 1, 10, 1000], 'nb__alpha':[0.01, 0.1, 1, 10, 1000]}

def new_func(pipeline, parameters):
    clf = GridSearchCV(pipeline, parameters, cv = 10, scoring="accuracy")
    return clf

clf = new_func(pipeline, parameters) 
clf.fit(X_train, y_train)

clf.best_params_

"""Save the file into an sav"""
filename = 'nb_cv.sav'
pickle.dump(clf, open(filename, 'wb'))

"""Load the model file"""
filename = 'nb_cv.sav'
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

