import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#nltk.download("stopwords")
#nltk.download('omw-1.4')


df = pd.read_csv('mexican_reviews.csv')
Food_lexicons = pd.read_csv('data/food.csv',header=None,names=['food']) 

FOOD_LEXICONS = Food_lexicons.food.values.tolist()
FOOD_LEXICONS = set([x.lower() for x in FOOD_LEXICONS])


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
freq= pd.Series(" ".join(df['review_nopunc_nostop']).split()).value_counts()[:50]
pd.set_option('display.max_rows', 500)
print(freq)

# Remove other necessary stop words
other_stopwords = ['get','also', 'back','us','go','would','even','went','say','day','ive','im']
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
df2.to_csv("artifacts/sentiment_score_by_review.csv",index=False)

# Polarity by restaurant
Rest_score = pd.DataFrame()
Rest_score['Restaurant_name'] = df.groupby(['restaurant_name','id'])['restaurant_name'].unique()
Rest_score['Polarity'] = df.groupby(['restaurant_name','id'])['polarity'].mean()
Rest_score.to_csv("artifacts/sentiment_score_by_rest.csv",index=False)



# Extract food from reviews based on food lexicon file
from nltk.stem.porter import PorterStemmer


def _extract_ngrams(data: str, num: int):
    '''
    function to return different successive parts of sentence
    '''
    n_grams = TextBlob(data).ngrams(num)
    return [' '.join(grams).lower() for grams in n_grams]

def _delete_duplicate_food_n_grams(text: str, foods: list[str]) -> list[str]:
    '''
    function to delete duplicate food items
    '''
    foods.sort(key=lambda x: -len(x.split()))  # Sort desc by number of words
    result_foods = []
    for food in foods:
        if food in text:
            text = text.replace(food, '')
            result_foods.append(food)
    return result_foods

def extract_foods(x: str) -> list[str]:
    '''
    function to extract food words from text based on comprehensive food lexicon list
    '''
    foods = set()
    stemmer = PorterStemmer()
    for n in range(6, 0, -1):
        n_grams = _extract_ngrams(x.cleaned_review, n)
        n_grams_stemmed = [stemmer.stem(n_gram) for n_gram in n_grams]
        n_grams_set = set(n_grams).union(n_grams_stemmed)
        foods = foods.union(n_grams_set.intersection(FOOD_LEXICONS))
    foods = list(foods)
    foods = _delete_duplicate_food_n_grams(x.cleaned_review, foods)
    return list(foods)


df['food'] = df.apply(extract_foods,axis =1)

# To choose categories for entity analysis
food_list = df.food.values.tolist()

food_dict = {}

for words in food_list:
    for word in words:
        if word in food_dict:
            food_dict[word] += 1
        else:
            food_dict[word] = 1
            
food_counts_list = [(k, v) for k, v in food_dict.items()]
sorted_food_counts = sorted(food_counts_list,key=lambda i: -i[1])

# WordCloud for reviews
cleaned_review_raw = df.cleaned_review.values.tolist()
cleaned_review_raw2 = [item for sublist in cleaned_review_raw for item in sublist]
cleaned_review_list = ''.join(cleaned_review_raw2)

wordcloud_reviews = WordCloud(width = 3000, height = 2000,random_state=1).generate(cleaned_review_list)
plt.imshow(wordcloud_reviews)
plt.axis("off")
plt.show()

# WordCloud for food

wordcloud_food = WordCloud(width = 3000, height = 2000,random_state=1).generate_from_frequencies(food_dict)
plt.imshow(wordcloud_food)
plt.axis("off")
plt.show()

