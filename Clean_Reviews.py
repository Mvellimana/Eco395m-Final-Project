import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob

#nltk.download("stopwords")
#nltk.download('omw-1.4')

os.chdir('/Users/Mai/Documents/ECO385M/Eco395m-Final-Project-jordan/artifacts') #update directory
df = pd.read_csv('yelp_all_reviews_combined.csv') 

df.shape

df.info()

print(len(df['restaurant_name'].unique()))

# Lower case all words
df['review_lower'] = df['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Remove Punctuation
df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]', '')

stop_words = stopwords.words('english')

# Remove Stopwords
df['review_nopunc_nostop'] = df['review_nopunc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

# Return frequency of values
freq= pd.Series(" ".join(df['review_nopunc_nostop']).split()).value_counts()[:100]
pd.set_option('display.max_rows', 500)
print(freq)

# Remove other necessary stop words
other_stopwords = ['get','also', 'back','us','go','would','even','went','say','day']
df['review_nopunc_nostop_nocommon'] = df['review_nopunc_nostop'].apply(lambda x: "".join(" ".join(x for x in x.split() if x not in other_stopwords)))

# Lemmatization

# Lemmatize final review format
df['cleaned_review'] = df['review_nopunc_nostop_nocommon'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()

# Basic Sentiment Analysis

# Calculate polarity
df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])

# Calculate subjectivity
df['subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])

# Filter neceassry columns for csv
df2 = df.loc[:,['restaurant_name','number','cleaned_review','polarity','subjectivity']]
df2.to_csv("Cleaned_reviews.csv",index=False)

# Polarity and Subjectivety by restaurant
df2.groupby(['restaurant_name'])['polarity'].mean()
df2.groupby(['restaurant_name'])['subjectivity'].mean()
