## All Predictive Models ##

"""Models
# 1. Naïve Bayes: Count Vectorizer 
# 2. Naïve Bayes: TF IDF 
# 3. Gradient Boosted Classifier: Count Vectorizer 
# 4. Gradient Boosted Classifier: TF IDF"""

"""Import Packages"""
import pandas as pd
import pickle
import sklearn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix


"""Import the Dataset"""
df = pd.read_csv('mexican_reviews.csv')

"""Drop rows with no ratings. """
yelp_d = df[df.rating != 'no rating']

"""One-hot-encoding. 
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
yelp_d['rating_binary'] = yelp_d.apply(star_rating,axis = 1)

"""Create two new dataframes to hold all the positive and negative samples. We do this because the positive samples are
much higher than the nehative samples. So we randomly sample the no. of positive reviews to match the no. of negative reviews. 
We then create a new dataframe to hold these 2 dfs (using concatenate)."""
df_positive = yelp_d[yelp_d['rating_binary'] == 1]
df_positive_sample = df_positive.sample(n=21386)

df_negative = yelp_d[yelp_d["rating_binary"] == 0]
df_negative_sample = df_negative.sample(n=21386)

yelp_data = pd.concat([df_positive_sample, df_negative_sample], axis=0)


"""Train-test split + random state"""   
X = yelp_data.review
y = yelp_data.rating_binary
indices = yelp_data.index

X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(X, y, indices, train_size = 0.8, random_state = 7)
print(X_train)


# Naive Bayes: Count Vectorizer

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


# Naive Bayes: TF IDF

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

nb_tf_acc = test_accuracy
nb_tf_f1 = f1_accuracyw
nb_tf_f1m = f1_accuracym
nb_tf_f1w = f1_accuracyw

# Gradient Boosted Classifier: Count Vectorizer

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

# Gradient Boosted Classifier: TF IDF

"""Model Validation"""
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
filename = 'gbc_tf.sav'
pickle.dump(clf, open(filename, 'wb'))

"""Load the model file"""
filename = 'gbc_tf.sav'
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

gbc_tf_acc = test_accuracy
gbc_tf_f1 = f1_accuracy
gbc_tf_f1m = f1_accuracym
gbc_tf_f1w = f1_accuracyw

# Common Results 

"""Aggregate Results"""

result2 = pd.DataFrame({'Model':['NB_CV', 'NB_TF', 'GBC_CV', 'GBC_TF'],
             'Accuracy':[nb_tf_acc,nb_tf_acc, gbc_cv_acc, gbc_tf_acc],
             'F1_Macro':[nb_tf_f1, nb_tf_f1, gbc_cv_f1, gbc_tf_f1],
             'F1_Micro':[nb_tf_f1m, nb_tf_f1m, gbc_cv_f1m, gbc_tf_f1m],
             'F1_Weighted':[nb_tf_f1w, nb_tf_f1w, gbc_cv_f1w, gbc_tf_f1w]})
result2 = result2.round(3)
result2

"""Confusion Matrix"""
cm = confusion_matrix(y_test, clf.predict(X_test), labels=None, sample_weight=None)
cm

cm_yelp_data =  pd.DataFrame(cm, index= [i for i in ['Negative','Positive']],
                     columns= [i for i in ['Negative','Positive']])

"""Heatmap"""
sns.heatmap(cm_yelp_data, annot=True,cmap='Blues',fmt='g')

