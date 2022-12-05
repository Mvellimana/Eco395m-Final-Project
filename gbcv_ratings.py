## Predictive Modelling [NLP]

# Gradient Boosting Classifier: Count Vectorizer

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
For all ratings <= 3, rating_binary = 0. For all ratings >= 4, rating_binary = 1. """
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

steps = [('vec', TfidfVectorizer(stop_words = 'english', ngram_range = (1, 2))), 
         ('gbc', GradientBoostingClassifier(max_features='sqrt',n_estimators=500))] 
pipeline = Pipeline(steps) 
parameters = {'gbc__learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25]}

clf = GridSearchCV(pipeline, parameters, cv = 3, scoring="accuracy") 
clf.fit(X_train, y_train)

clf.best_params_

steps = [('vec', TfidfVectorizer(stop_words = 'english', ngram_range = (1, 2))), 
         ('gbc', GradientBoostingClassifier(learning_rate = 0.25, max_features = 'sqrt', n_estimators = 500))] 
clf = Pipeline(steps) 
clf.fit(X_train, y_train)

"""Save the file into an sav"""
filename = 'gbc_cv.sav'
pickle.dump(clf, open(filename, 'wb'))

"""Load the model file"""
filename = 'gbc_cv.sav'
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

gbc_cv_acc = test_accuracy
gbc_cv_f1 = f1_accuracy
gbc_cv_f1m = f1_accuracym
gbc_cv_f1w = f1_accuracyw