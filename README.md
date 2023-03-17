# Opinions Mining of Yelp Reviews

Please refer to the link below to our streamlit Dashboard.
https://mvellimana-eco395m-final-pro-code-streamlitstreamlit-app-63u55v.streamlit.app/


## Introduction
This project aims to conduct an opinions mining analysis based on yelp reviews. The projects includes the use of Google Cloud Platform, SQL, scraping, sentiment analysis, and Streamlit. We were able to view the results of the sentiment analysis based on yelp reviews.

## Data Source
### I. API Scrapping
To start this project, we began with using Yelp’s Fusion API system to gather business ID’s as well as a JSON file associated with those IDs. This JSON file contained information about the restaurant such as the name of the restaurant, number of reviews, ratings, location, etc. We followed up this by then using the ID to also gather up to three reviews per restaurant (Yelp’s API limit). Sadly Yelp also limited the maximum number of characters in a review. Overall, this limited the possibilities of what we could do with sentiment analysis. That’s when a beacon was lit and Super Scraper Jordan came to our rescue. Using web scraping, we were able to come up with essentially all the reviews for each Mexican restaurant in Austin.

### II. GCP Database
For our database we used a PostgreSQL 13 server on the Google Cloud Platform with the following summary:

Picture1.png

Which in hindsight might have been a little more powerful than we needed but the server was still cheaper than expected.

## Data Exploration
In order to analyze the data, we conducted some data exploration to give some interactive insights to the user of this streamlit.

### I. Restaurants Map
First, we were able to plot the locations of all the Mexican restaurants in Austin on a map with the ability to filter them according to the ratings. It gives the user first impressions of where the best Mexican restaurants are in Austin.

### II. Word Clouds
Since we were able to scrape all the reviews for all the Mexican restaurants in Austin, we were able to come up with the top mentioned or iterated words. Our focus was on the top words iterated in general within all the scraped reviews, and another specific cloud for the most dishes mentioned in those reviews. We used wordcloud package in order to produce that.

### III. Restaurants Filter APP
In this section we were able to create 2 interactive apps that filter restaurants based on different preferences. The first app uses the restaurants info scraped data to allow the user to filter by 4 different attributes. On the other hand, the second app filter the restaurants based on price and ratings. Both apps provide unique info to the user.

## Methdology
Online reviews have the power to drive customers to or away from a business, and tell you what customers like and dislike about it. We performed sentiment and entity analysis on restaurant reviews to uncover emotions in online reviews. Both analysis detect trends and patterns that may not be evident at first glance. We also tested the accuracy of the sentiment score using TextBlob pre-built model and a Hand-Built Naïve Bayes TF IDF (Bag-of-Words) Model. Our approaches are elaborated in further detail in the sections below.

### I. Sentiment Analysis
Sentiment Analysis provides an effective way to evaluate the sentiment behind text. It typically uses machine learning algorithms and natural language processing (NLP) to estimate whether the data is positive, neutral or negative.

#### Review Sentiment Analysis Overview
We used the sentiment property from the TexBlob package to approximate whether the restaurant review is favorable or not. TextBlob is a python library to process textual data. The sentiment function returns text polarity that ranges from -1 to 1 indicating if the text is negative or positive. TextBlob is a rule-based sentiment analyzer that implements a lexicon based approach. Here a sentiment is defined by its semantic orientation and the intensity of each word in the sentence. This will use a pre-defined dictionary classifying negative and positive words. Generally, a text message will be represented by bag of words. Initially individual scores are assigned to all the words after which a final sentiment is calculated by some pooling operation.

#### Data Cleaning
To use this module, we firstly had to make sure that our reviews are in the correct format. This involved some data cleaning. The main steps for the data cleaning part are :

1. Lowercase all words: To insure, consistency with list of stopwords, as well as to include capitalized words
2. Remove punctuation
3. Remove stop words: Some commonly occuring words have little to no meaning and removing them from the data makes sentiment analysis more efficent. The Natural Language Toolkit comes built in with a comprehensive stop words list. This list, which was imported from nltk.corpus library, was used to remove stop words from the Yelp reviews. Further removal was done by picking out trivial words from a visual check of top 50 most frequent occurring words assigned to all the words after which a final sentiment is calculated by some pooling operation.
4. Lemmatization: Lemmatization is basically a method that switches any kind of a word to its base root mode. This cuts out the number of words that are available for analysis by combining similar forms into one base form.

### 1. TextBlob Algorothim - Reviews Polarity Score
Once the reviews were cleaned and formatted, the sentiment property from TextBlob was applied to every review to get its polarity. This polarity scores were averaged by each unique restaurant to get a restaurant level sentiment score.

#### Limitations
- Averaging sentiment scores of all reviews by restaurant might not give an accurate representation of the restaurant’s reputation.

- Textblob ignores words that it does not know and will only consider words and phrases that it can assign polarity to. This could skew the polarity results if unknown words are many.

### 2. Google Cloud NLP Algorothim - Entity Sentiment Analysis
It would be interesting to find what the customers feel about certain aspects of importance from the reviews. That is if multiple reviews talk about tacos then what do yelpers in general feel about the tacos there (is it a must-go or must-avoid). We use Google Cloud NLP sentiment analysis here. Being one of the largest data aggregators of some form, Google is in an unparalleled position to perform some NLP exercises. Entity sentiment analyser in GCP is built to identify entities from large texts and also identify whether that entity is in positive or negative standing with respect to the whole review. It is also smart enough to know if multiple words/phrases refer to the same entity. It generates a lot of useful information like salience score, sentiment score, sentiment magnitude, number of mentions etc.

- Salience score tells us the relevance of the entity to the review
- Sentiment score tells us what the reviewer feels about the entity
- Magnitude tells us how emotional the review is (does it seem to be written by a bot ?)

However, for the sake of this project, we focused only on sentiment score.

Since our focus is on Mexican restaurants and its reviews by Yelpers. We looked at roughly 70k reviews which is a rich information for any form of text analysis. To avoid maxing out on free GCP credits, we emphasized on 10 popular food items from Mexican cuisine. We chose this list based on word cloud and n-grams frequency. The goal of this analysis is to get the top and worst 5 restaurants for each enetity (food item) according to the sentiment score results. The entity categories are:

1. Tacos
2. Tortillas
3. Chicken
4. Enchilada
5. Fajita
6. Burrito
7. Fish
8. Shrimp
9. Quesadilla
10. Steak

Post this filtering we end up with 45000 reviews which is still huge but a significant reduction. The one thing that connects the code to your GCP account is the API Key (which must be kept invisible to avoid fraudulent loss of credits). We then study the reviews to arrive at an entity sentiment score pivoted around the above-mentioned 10 categories and provide recommendations for top 5 and bottom 5 restaurants for a particular food item.

Further explaination on how we established the popular food categories, we used a comprehensive domain specific food items lexicon corpus which is available in YelpNLG: Review Corpus for NLG. This lexicon library was used to extract food items from reviews. Through word cloud we knew the most frequent words/phrases. So we mapped and compared these two and identified only popular food items in the reviews. The only shortcoming we could think of for GCP NLP is that it's not free and not cheap as we do more analysis. They have very specific billing regulations and one must be careful especially while doing high computational analysis.

### II. Sentiment Analysis Validations: The Use of Predictive Analytics

#### Objective
In this section, we have built classification models to help us gauge the accuracy with which our model is classifying an event as a true positive or negative. It provides a sanity check for the unsupervised TextBlob’ pre-built model by predicting its accuracy. On top of that, we created a hand-built model to compare its accuracy with the pre-built one. So, we offer two approaches to test the accuracy of our classification.

#### Approach 1: Pre-Built Model
We used a Custom-Built Sentiment Analyzer using Textblob Naïve Bayes Classifier. The highlights of the model are as follow:

- Textblob is a pre-built deep learning model that is trained on movie reviews. Hence, it might not work accurately for food reviews, leading to biased results.
- We used TextBlob to find the polarity score of the food reviews. To predict the accuracy of this polarity score, we built a sentiment analyzer using TextBlob’s Naïve Bayes classifier.
- We used the polarity score (where 0 = negative, 1 = positive) to build the model.
- This analyzer model achieved an accuracy of 89.69%.

#### Approach 2: Hand-Built Model
In this model, we created a Hand-Built Naïve Bayes TF IDF (Bag-of-Words) Model. The highlights as follow:

- We built a model from scratch to predict the accuracy with which a review is correctly classified as positive.
- We used ratings to classify the review as positive or negative. We use this as a proxy for the polarity score as that is an inferred score. Ratings, on the other hand, are provided by the individual writing the review, and so can be considered as ground data.
- We classify 1, 2 and 3-star ratings as negative (0) and 4 and 5-star ratings as positive (1) to build the model.
- This model achieved an accuracy of 83.8%.

##### Winning Model
Based on our predictions, we selected the TextBlob algorithm as it provides higher accuracy. However, we must consider the fact that this model builds on the polarity score that the pre-built model produces. So, if a review has wrongly been estimated as positive, then the algorithm might also end up classifying it as positive. However, in the interest of time and efficiency, we are looking at only accuracy. To get better results, we can look at metrics like the precision and recall scores, the F1 scores and the ROC-AUC Curve.

##### Note
We had built 4 hand-built models to test for the one that gave the best accuracy.

1. Naïve Bayes: Count Vectorizer
2. Naïve Bayes: TF IDF
3. Gradient Boosted Classifier: Count Vectorizer
4. Gradient Boosted Classifier: TF IDF

While all of them gave nearly identical results, the Naïve Bayes TF IDF model performed slightly better than the other three. Hence, we picked that one as the best model for Approach 2. Most of the yelp reviews were positive, which caused the model to give biased results. To eliminate this imbalance, we reduced the number of positive reviews in the datasets to match the negative reviews.

### III. Streamlit App
We use streamlit to produce a dashboard to provide connections with users who have the need to find a potential interested Mexican restaurant. We used streamlit package as well as some plot and animation packages to beautify the dashboard. The dashboard provides final results of the analysis of the scraped data, and the different sentiment analysis. It contains an interactive apps such as dashboards that provide filtration functions for a user to find his/her interested restaurant.

## Results and Analysis

All of the results of the sentiment analysis is shown in the “Sentiment Analysis” page in this streamlit app. It starts by showing the accuracy comparison between the two different models. Then, we created an interactive dashboard that allow the user to select a restaurant based on filtration function that utilizes the sentiment analysis results of the TextBlob which is the winning model according to our predictive analysis. Finally, it shows another dashboard that provide the top and worst 5 restaurants according to each entity (food item). This dashboard is based on the Google Cloud NLP - Entity Sentiment Analysis results.

## Reproducibility

### Data Scrapping
To reproduce our results, first go make sure to create a .env file under the code_Yelp_API directory with the credentials to access your database. An example, demo.env, is in the same directory. Then, run code_Yelp_API/scrape_businesses.py to gather the business ID and JSON file associated with it. Then, if you only want up to three reviews you can run code_Yelp_API/scrape_reviews.py. In order to gather all the reviews, we first need to navigate to the code(scraping) directory and install the proper packages in Python. We can do this by running pip install -r requirements.txt. After our packages are installed, we must create a .env file in the code(scraping) directory with the database’s credentials. Once this is done, we can now access the database and run the code. First, restaurant_info_from_db_to_csv.py will grab all the restaurant information from the database. Then, scrape_restaurant_attributes.py and scrape_reviews.py will access the restaurants’ urls from the database to scrape their Yelp page and get attributes, reviews, and other information included in the review section, like reviewer’s elite status. Lastly, you run clean_reveiws.py, which will produce a version of the reviews without any special characters. This is necessary for some of the sentiment analysis.

### Reviews Sentiment Score
For the sentiment score for each restaurant, the sentiment_score_&_clean_reviews.py script under code_sentiment_analysis directory needs to be executed. This will save two csv files, one with sentiment score per review and the other one with sentiment score per restaurant, under the artifacts directory.

### Entity Sentiment Score
To reproduce the entity sentiment score, you will require your own API key (API keys we used have not been provided in the repo). We do want to highlight that due to large computing time we had to save the data from API calls which were too big to be pushed through git. There was no real reason for saving the outputs except for safety and to avoid overuse of credits. From there on the codes in the code_sentiment_analysis directory can be followed and executed.

### Sentiment Analysis Validations
For the classification models, two different CSVs have been used. The TextBlob Analyzer uses the sentiment_score_by_review.csv and The Naïve Bayes TF IDF uses the mexican_reviews.csv. The codes can be found in the code_sentiment_analysis directory with names allpredmodels.py and textblobanalyzer.py. On the other hand, the csv files could be found in the artifacts directory.

### Streamlit App
In order to produce the streamlit app. We have used different types of packages which could be found in the requirments.txt under the code_streamlit directory. In addition, the code could be found in the same directory with the name of streamlit_app.py. It’s worth mentioning that we have deployed the app by creating an account in streamlit and connecting the Github page with the account.

