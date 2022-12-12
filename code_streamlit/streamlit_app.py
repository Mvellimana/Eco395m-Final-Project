import ast
import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import os


# Preparing 5 pages for the app in the sidebar
#Create a path for sidebar picture

PATH1 = os.path.join("artifacts", "yelp_logo.png")
PATH_polarity = os.path.join("artifacts", "polarity.png")

with st.sidebar:
	st.image(PATH1)
	choice=option_menu(
		menu_title="Final Project",
		options=["Overview","Restaurants Map", "Top Mentioned Words","Choose Your Restaurants", "Sentiment Analysis"],
		icons=["house", "map", "book", "list-task", "list"],
		menu_icon="cast",
		default_index=0,
		)


#load function to add animation effect

def load_lottiefile(filepath: str):
	'''load lottifile to add animation effects.'''
	with open(filepath, "r") as f:
		return json.load(f)


#Build the first page 

if choice=="Overview":

	st.title("Opinions Mining of Yelp Reviews")
	
	st.image(PATH_polarity)

	st.header("Introduction")
	st.markdown("This project aims to conduct an opinions mining analysis based on yelp reviews. The projects includes the use of Google Cloud Platform, \
		SQL, scraping, sentiment analysis, and Streamlit. We were able to view the results of the sentiment analysis based on yelp reviews.")


	st.header("Data Source")
	st.subheader("I. API Scrapping")
	st.markdown("To start this project, we began with using Yelp’s Fusion API system to gather business ID’s as well as a JSON file associated with those IDs.\
	 This JSON file contained information about the restaurant such as the name of the restaurant, number of reviews, ratings, location, etc. \
	 We followed up this by then using the ID to also gather up to three reviews per restaurant (Yelp’s API limit). \
	 Sadly Yelp also limited the maximum number of characters in a review. Overall, this limited the possibilities of what we could do with sentiment analysis. \
	 That’s when a beacon was lit and Super Scraper Jordan came to our rescue. \
	 Using web scraping, we were able to come up with essentially all the reviews for each Mexican restaurant in Austin.")

	st.subheader("II. GCP Database")
	st.markdown("For our database we used a PostgreSQL 13 server on the Google Cloud Platform with the following summary:")
	st.markdown("[![Picture1.png](https://i.postimg.cc/SsrDBzt4/Picture1.png)](https://postimg.cc/XGrfFXb1)")

	st.markdown("Which in hindsight might have been a little more powerful than we needed but the server was still cheaper than expected.")


	st.header(" Data Exploration")
	st.markdown("In order to analyze the data, we conducted some data exploration to give some interactive insights to the user of this streamlit.")
	st.subheader("I. Restaurants Map")
	st.markdown("First, we were able to plot the locations of all the Mexican restaurants\
	 in Austin on a map with the ability to filter them according to the ratings. \
	 It gives the user first impressions of where the best Mexican restaurants are in Austin.")


	st.subheader("II. Word Clouds")
	st.markdown("Since we were able to scrape all the reviews for all the Mexican restaurants in Austin, we were able to come up with the top mentioned or iterated words.\
	 Our focus was on the top words iterated in general within all the scraped reviews, and another specific cloud for the most dishes mentioned in those reviews.\
	  We used `wordcloud` package in order to produce that.")

	st.subheader("III. Restaurants Filter APP")
	st.markdown("In this section we were able to create 2 interactive apps that filter restaurants based on different preferences. \
		The first app uses the restaurants info scraped data to allow the user to filter by 4 different attributes. \
		On the other hand, the second app filter the restaurants based on price and ratings. Both apps provide unique info to the user.")


	st.header("Methdology")
	st.markdown("Online reviews have the power to drive customers to or away from a business, and tell you what customers like and dislike about it.\
	 We performed sentiment and entity analysis on restaurant reviews to uncover emotions in online reviews. Both analysis detect trends and patterns that \
	 may not be evident at first glance. We also tested the accuracy of the sentiment score using `TextBlob` pre-built model and a Hand-Built `Naïve Bayes TF IDF` \
	 (Bag-of-Words) Model. Our approaches are elaborated in further detail in the sections below.")

	st.subheader("I. Sentiment Analysis")
	st.markdown("Sentiment Analysis provides an effective way to evaluate the sentiment behind text. \
		It typically uses machine learning algorithms and natural language processing (NLP) to estimate whether the data is positive, \
		neutral or negative.")
 
	st.markdown("##### Review Sentiment Analysis Overview")


	st.markdown("We used the sentiment property from the `TexBlob` package to approximate whether the restaurant review is favorable or not.\
 	`TextBlob` is a python library to process textual data. The sentiment function returns text polarity that ranges from -1 to 1 indicating if the text is negative\
  	or positive. `TextBlob` is a rule-based sentiment analyzer that implements a lexicon based approach. Here a sentiment is \
  	defined by its semantic orientation and the intensity of each word in the sentence. This will use a pre-defined dictionary \
  	classifying negative and positive words. Generally, a text message will be represented by bag of words. Initially individual scores are \
  	assigned to all the words after which a final sentiment is calculated by some pooling operation.")


	st.markdown("##### Data Cleaning")

	st.write('''
	To use this module, we firstly had to make sure that our reviews are in the correct format. \
	This involved some data cleaning. The main steps for the data cleaning part are :''')

	st.markdown("1. Lowercase all words: To insure, consistency with list of stopwords, as well as to include capitalized words")
	
	st.markdown("2. Remove punctuation")
	st.markdown("3. Remove stop words: Some commonly occuring words have little to no meaning and removing them from the data makes sentiment analysis more efficent. \
	The Natural Language Toolkit comes built in with a comprehensive stop words list. This list, which was imported from `nltk.corpus` library, was used to \
	remove stop words from the Yelp reviews. Further removal was done by picking out trivial words from a visual check of top 50 most frequent occurring words\
	assigned to all the words after which a final sentiment is calculated by some pooling operation.")

	st.markdown("4. Lemmatization: Lemmatization is basically a method that switches any kind of a word to its base root mode. This cuts out the number of words that \
	are available for analysis by combining similar forms into one base form.")

	st.markdown("#### 1. TextBlob Algorothim - Reviews Polarity Score")

	st.markdown("Once the reviews were cleaned and formatted, the sentiment property from `TextBlob` was applied to every review to get its polarity. \
	This polarity scores were averaged by each unique restaurant to get a restaurant level sentiment score.")


	st.markdown("##### Limitations")

	st.markdown(" - Averaging sentiment scores of all reviews by restaurant might not give an accurate representation of the restaurant’s reputation.")
	st.markdown(" - `Textblob` ignores words that it does not know and will only consider words and phrases that it can assign polarity to. \
	This could skew the polarity results if unknown words are many.")

	st.markdown("#### 2. Google Cloud NLP Algorothim - Entity Sentiment Analysis")

	st.markdown("It would be interesting to find what the customers feel about certain aspects of importance from the reviews. \
	That is if multiple reviews talk about tacos then what do yelpers in general feel about the tacos there (is it a must-go or must-avoid).\
	 We use `Google Cloud NLP` sentiment analysis here. Being one of the largest data aggregators of some form, Google is in an unparalleled position to \
	 perform some NLP exercises. Entity sentiment analyser in `GCP` is built to identify entities from large texts and also identify whether that entity\
	  is in positive or negative standing with respect to the whole review. It is also smart enough to know if multiple words/phrases refer to the same entity. \
	  It generates a lot of useful information like salience score, sentiment score, sentiment magnitude, number of mentions etc.")
	st.markdown(" - Salience score tells us the relevance of the entity to the review")
	st.markdown(" - Sentiment score tells us what the reviewer feels about the entity")
	st.markdown(" - Magnitude tells us how emotional the review is (does it seem to be written by a bot ?)")
	st.markdown(" However, for the sake of this project, we focused only on sentiment score.")


	st.markdown("Since our focus is on Mexican restaurants and its reviews by Yelpers. We looked at roughly 70k reviews which is a rich information \
	for any form of text analysis. To avoid maxing out on free GCP credits, we emphasized on 10 popular food items from Mexican cuisine. \
	We chose this list based on word cloud and n-grams frequency. The goal of this analysis is to get the top and worst 5 restaurants for each enetity (food item)\
	according to the sentiment score results. The entity categories are:")

	st.markdown("1. Tacos")
	st.markdown("2. Tortillas")
	st.markdown("3. Chicken")
	st.markdown("4. Enchilada")
	st.markdown("5. Fajita")
	st.markdown("6. Burrito")
	st.markdown("7. Fish")
	st.markdown("8. Shrimp")
	st.markdown("9. Quesadilla")
	st.markdown("10. Steak")

	st.markdown("Post this filtering we end up with 45000 reviews which is still huge but a significant reduction. \
	The one thing that connects the code to your GCP account is the API Key (which must be kept invisible to avoid fraudulent loss of credits).\
	We then study the reviews to arrive at an entity sentiment score pivoted around the above-mentioned 10 categories and provide recommendations\
	for top 5 and bottom 5 restaurants for a particular food item.")

	st.markdown("Further explaination on how we established the popular food categories, we used a comprehensive domain specific food items\
 	*lexicon corpus* which is available in `YelpNLG: Review Corpus for NLG`. This lexicon library was used to extract food items from reviews. \
 	Through word cloud we knew the most frequent words/phrases. So we mapped and compared these two and identified only popular food items in the reviews.\
	The only shortcoming we could think of for `GCP NLP` is that it's not free and not cheap as we do more analysis. \
 	They have very specific billing regulations and one must be careful especially while doing high computational analysis.")


	st.subheader("II. Sentiment Analysis Validations: The Use of Predictive Analytics")

	st.markdown("#### Objective")

	st.markdown("In this section, we have built classification models to help us gauge the accuracy with which our model is classifying\
 	an event as a true positive or negative. It provides a sanity check for the unsupervised `TextBlob’` pre-built model by predicting its accuracy.\
 	On top of that, we created a hand-built model to compare its accuracy with the pre-built one. So, we offer two approaches to test the accuracy of our classification.")

	st.markdown("#### Approach 1: Pre-Built Model")

	st.markdown("We used a Custom-Built Sentiment Analyzer using `Textblob Naïve Bayes Classifier`. The highlights of the model are as follow:")
	st.markdown(" - `Textblob` is a pre-built deep learning model that is trained on movie reviews. Hence, it might not work accurately for food reviews,\
 	leading to biased results.")
	st.markdown(" - We used `TextBlob` to find the polarity score of the food reviews. To predict the accuracy of this polarity score,\
 	we built a sentiment analyzer using `TextBlob’s Naïve Bayes classifier`.")
	st.markdown("- We used the polarity score (where 0 = negative, 1 = positive) to build the model.")
	st.markdown("- This analyzer model achieved an accuracy of 89.69%.")


	st.markdown("#### Approach 2: Hand-Built Model")

	st.markdown("In this model, we created a Hand-Built `Naïve Bayes TF IDF` (Bag-of-Words) Model. The highlights as follow:")
	st.markdown(" -  We built a model from scratch to predict the accuracy with which a review is correctly classified as positive.")
	st.markdown(" -  We used ratings to classify the review as positive or negative. We use this as a proxy for the polarity score as that is an inferred score.\
 	Ratings, on the other hand, are provided by the individual writing the review, and so can be considered as ground data.")
	st.markdown(" - We classify 1, 2 and 3-star ratings as negative (0) and 4 and 5-star ratings as positive (1) to build the model.")
	st.markdown("- This model achieved an accuracy of 83.8%.")

	st.markdown("##### Winning Model")
	st.markdown("Based on our predictions, we selected the `TextBlob` algorithm as it provides higher accuracy. \
	However, we must consider the fact that this model builds on the polarity score that the pre-built model produces. \
	So, if a review has wrongly been estimated as positive, then the algorithm might also end up classifying it as positive. \
	However, in the interest of time and efficiency, we are looking at only accuracy. To get better results, we can look at metrics \
	like the precision and recall scores, the F1 scores and the ROC-AUC Curve.")

	st.markdown("##### Note")
	st.markdown("We had built 4 hand-built models to test for the one that gave the best accuracy. ")
	st.markdown("1. `Naïve Bayes: Count Vectorizer`")
	st.markdown("2. `Naïve Bayes: TF IDF`")
	st.markdown("3. `Gradient Boosted Classifier: Count Vectorizer`")
	st.markdown("4. `Gradient Boosted Classifier: TF IDF`")

	st.markdown("While all of them gave nearly identical results, the Naïve Bayes TF IDF model performed slightly better than the other three. \
	Hence, we picked that one as the best model for Approach 2. Most of the yelp reviews were positive, which caused the model to give biased results.\
	 To eliminate this imbalance, we reduced the number of positive reviews in the datasets to match the negative reviews.")

	st.subheader("III. Streamlit App")
	st.markdown("We use `streamlit` to produce a dashboard to provide connections with users who have the need to find a potential interested Mexican restaurant.\
 	We used `streamlit` package as well as some plot and animation packages to beautify the dashboard. The dashboard provides final results of \
 	the analysis of the scraped data, and the different sentiment analysis. It contains an interactive apps such as dashboards that provide \
 	filtration functions for a user to find his/her interested restaurant.")

	st.header("Results and Analysis")
	st.markdown("All of the results of the sentiment analysis is shown in the *“Sentiment Analysis”* page in this streamlit app. \
	It starts by showing the accuracy comparison between the two different models. Then, we created an interactive dashboard that allow \
	the user to select a restaurant based on filtration function that utilizes the sentiment analysis results of the `TextBlob` \
	which is the winning model according to our predictive analysis. Finally, it shows another dashboard that provide the top and worst 5 restaurants according \
	to each entity (food item). This dashboard is based on the `Google Cloud NLP - Entity Sentiment Analysis` results.")

	st.header("Reproducibility")
	st.subheader("Data Scrapping")
	st.markdown("To reproduce our results, first go make sure to create a `.env` file under the `code_Yelp_API` directory with the credentials to access your database.\
 	An example, `demo.env`, is in the same directory. Then, run `code_Yelp_API/scrape_businesses.py` to gather the business ID and JSON file associated with it. \
 	Then, if you only want up to three reviews you can run `code_Yelp_API/scrape_reviews.py`. \
 	In order to gather all the reviews, we first need to navigate to the code(scraping) directory and install the proper packages in Python. \
 	We can do this by running `pip install -r requirements.txt`. After our packages are installed, we must create a `.env` file in the `code(scraping)` \
 	directory with the database’s credentials. Once this is done, we can now access the database and run the code. \
 	First, `restaurant_info_from_db_to_csv.py` will grab all the restaurant information from the database. \
 	Then, `scrape_restaurant_attributes.py` and `scrape_reviews.py` will access the restaurants’ urls from the database to scrape their Yelp page and get attributes, \
 	reviews, and other information included in the review section, like reviewer’s elite status. Lastly, you run `clean_reveiws.py`, \
 	which will produce a version of the reviews without any special characters. This is necessary for some of the sentiment analysis.")

	st.subheader("Reviews Sentiment Score")
	st.markdown("For the sentiment score for each restaurant, the `sentiment_score_&_clean_reviews.py` script under `code_sentiment_analysis` directory needs to be executed.\
 	This will save two csv files, one with sentiment score per review and the other one with sentiment score per restaurant, under the `artifacts` directory. ")

	st.subheader("Entity Sentiment Score")
	st.markdown("To reproduce the entity sentiment score, you will require your own API key (API keys we used have not been provided in the repo). \
	We do want to highlight that due to large computing time we had to save the data from API calls which were too big to be pushed through git. \
	There was no real reason for saving the outputs except for safety and to avoid overuse of credits. \
	From there on the codes in the `code_sentiment_analysis` directory can be followed and executed.")

	st.subheader("Sentiment Analysis Validations")
	st.markdown("For the classification models, two different CSVs have been used. \
	*The TextBlob Analyzer* uses the `sentiment_score_by_review.csv` and *The Naïve Bayes TF IDF* uses the `mexican_reviews.csv`. \
	The codes can be found in the `code_sentiment_analysis` directory with names `allpredmodels.py` and `textblobanalyzer.py`. \
	On the other hand, the csv files could be found in the `artifacts` directory.")

	st.subheader("Streamlit App")
	st.markdown("In order to produce the streamlit app. We have used different types of packages which could be found in the `requirments.txt` under the `code_streamlit` \
	directory. In addition, the code could be found in the same directory with the name of `streamlit_app.py`. \
	It’s worth mentioning that we have deployed the app by creating an account in streamlit and connecting the Github page with the account. ")

	st.markdown("Big thanks to the contributers below:")
	st.markdown("**Ahmed Almezail**: [Github](https://github.com/Mezalay) [Linkedin](https://www.linkedin.com/in/ahmed-almezail)")
	st.markdown("**Jordan Despian**: [Github](https://github.com/JordanDespain) [Linkedin](https://www.linkedin.com/in/jordandespain)")
	st.markdown("**Lu Zhang**: [Github](https://github.com/MeetLuna)")
	st.markdown("**Maithreyi Vellimana**: [Github](https://github.com/Mvellimana) [Linkedin](https://www.linkedin.com/in/maithreyi-vellimana-e-i-t-2514a8115)")
	st.markdown("**Rajsitee Dhavale**: [Github](https://github.com/Rajsitee) [Linkedin](https://www.linkedin.com/in/rajsitee)")
	st.markdown("**Sonali Mishra**: [Github](https://github.com/sonali4794) [Linkedin](https://www.linkedin.com/in/smishra-xyz)")
	st.markdown("**Tyler Nicholson**: [Github](https://github.com/Tnich1995)")




#Build the path for the map data


PATH2 = os.path.join("artifacts", "yelp_map_data.csv")
PATH_MAP_ICON = os.path.join("artifacts", "map_icon.png")

map_df = (
	pd.read_csv(PATH2)
	)


if choice=="Restaurants Map":
	st.header("Locate the Highest Ratings Mexican Restaurants in Austin")

	st.image(PATH_MAP_ICON, )

	st.markdown('Choose the minimum rating.')

	rating = st.slider("Rating", 1.0, float(map_df["rating"].max()), 4.5, step = 0.5)
	st.map(map_df.query("rating >= @rating")[["latitude", "longitude"]])
	

#Build the path for the wordcloud pic


PATH3 = os.path.join("artifacts", "mexican_food.png")
PATH4 = os.path.join("artifacts", "wordcloud_food.png")
PATH5 = os.path.join("artifacts", "wordcloud_reviews.png")


#Design the 3rd page in the app

if choice=="Top Mentioned Words":

	st.title("Top Words/Dishes")
	st.image(PATH3)


	st.markdown("This analysis is conducted based on the highest iterated text from the scrapped yelp reviews on Mexican restaurants in Austin. \
		Click on the buttons below to explore them. :smile:")
	st.subheader("Top Iterated Dishes")
	

	if st.button("Top Iterated Dishes"):
		st.image(PATH4)

	st.subheader("Top Iterated Words Reviews")
	

	if st.button("Top Iterated Words Reviews"):
		st.image(PATH5)

	

#Design the 4th page 




PATH6 = os.path.join("artifacts", "mexican_restaurant_attributes.csv")
PATH7 = os.path.join("artifacts", "mexican_restaurant_info.csv")
PATH8 = os.path.join("artifacts", "food_choice.png")





#1st part of the page


if choice=="Choose Your Restaurants":
	st.title("Choose Your Restaurants")

	st.image(PATH8)
	st.subheader("Select your Favorite Restaurants Based on Attributes")

	attribute_choice=st.multiselect("Filter by attributes",["Takes Reservations", "Offers Delivery","Offers Takeout", "Masks required"])
    
	

	df_att = pd.read_csv(PATH6)
	
	

	def contain_selected_attributes(attributes, attribute_choice_set):
		"""Check if selected attributes are contained, if so, return True, otherwise, return False."""
		def convert_string_to_list(attributes):
			"""Helper function to convert attributes from string representation to list"""
			return ast.literal_eval(attributes)

		attributes_list = convert_string_to_list(attributes)
		return attribute_choice_set.issubset(set(attributes_list))


	selected_att = df_att["attributes"].apply(
	    contain_selected_attributes, args=(set(attribute_choice),)
	)

	st.dataframe(df_att[selected_att])
	
	
#2nd part of the page

	a,b=st.columns([3,1])
	money_choice=a.selectbox("Price Range",[" ","$$$","$$","$"]) 
	rating_choice = b.number_input("Minimum Rating", step=0.5)


	df_info = pd.read_csv(PATH7)

	st.subheader("Select your Favorite Restaurants Based on Price and Rating")

	def price_check(price, price_choice):
		'''get the boolean values for price.'''
		return price== price_choice

	if money_choice == ' ':
		price_bool = [True] * len(df_info.index)
	else:
		price_bool = df_info['price'].apply(price_check, args=(money_choice, ))

	
	def rating_check(rating, rating_input):
		'''get the boolean values for rating scores.'''
		return float(rating) >= float(rating_input)

	rating_bool = df_info['rating'].apply(rating_check, args=(rating_choice, ))

	bool_select = [a and b for a, b in zip(price_bool, rating_bool)]

	df_limit_cols = df_info[['name', 'review_count', 'rating', 'price', 'display_phone']]

	st.dataframe(df_limit_cols[bool_select])



#Build the last page app

PATH9 = os.path.join("artifacts", "sentiment_score_by_rest.csv")
PATH10 = os.path.join("artifacts", "best_rest.csv")
PATH11 = os.path.join("artifacts", "worst_rest.csv")	
PATH12 = os.path.join("artifacts", "sentiment.png")	


	#first part

if choice=="Sentiment Analysis":

	st.title("Sentiment Analysis")
	st.image(PATH12)


	st.header("Pre-Built Algorthim vs Hand-Built Algorthim")
	st.markdown("We have conducted 2 different models to assess the sentiment analysis. The first model is a **TextBlob** model which is a *Pre-Built Algorthim* that uses\
		reviews text as input. The other model is a **Naïve Bayes- TF IDF** which is a *Hand-Built Model* that uses rating as input which could be a proxy for reviewers sentiment.\
		The objective is to compare the accuracy of both models.")
	col1, col2 = st.columns(2)
	col1.metric("Pre-Built Algorthim", "89.67% Accuracy", "TextBlob")
	col2.metric("Hand-Built Algorthim", "83.8% Accuracy", "Naïve Bayes- TF IDF")
	st.markdown("So, we are using the **TextBlob** model results to show them below.")

	#2nd part
	
	df_polarity = pd.read_csv(PATH9)
	st.header("Find the Sentiment Score for Mexican Restaurants")
	st.markdown("Slide right for restaurants with higher sentiment analysis, and left to include the lower.")
	score=st.slider("Select the minimum sentiment score.",min_value=-1.0,max_value=1.0,value=0.00,step=0.001)
	df_polarity2 =df_polarity.loc[df_polarity["Polarity"]>=score]
	st.dataframe(df_polarity2)
	
	
	#3rd part

	st.header("Find the Best and the worst restaurants from the selectbox")
	st.markdown("Pick a specific dish to find the top 5 and worst 5 restaurants based on Google Cloud NLP.")
	df_best = pd.read_csv(PATH10)
	df_worst = pd.read_csv(PATH11)

	st.subheader("Top 5 restaurants per food item based on sentiment analysis 👍")
	dish = st.selectbox(
	    "Choose your fav dish",
	    [' ','tortilla', 'burrito', 'quesadilla', 'steak','fish', 'enchilada', 'fajita', 'taco', 'chicken', 'shrimp'])
	

	def food_best(food, food_choice):
	    return str(food_choice) == str(food)

	if dish == ' ':
	    food_bool = [False] * len(df_best.index)
	elif dish != ' ':
	    food_bool_best = df_best['cleaan_entity_name'].apply(food_best, args=(dish, ))
	    df_best2 = df_best[food_bool_best]
	    st.dataframe(df_best2)


	#Worst 5

	st.subheader("Worst 5 restaurants per food item based on sentiment analysis 👎")

	dish2 = st.selectbox(
	    "Choose your fav dish",
	    ['  ','tortilla', 'burrito', 'quesadilla', 'steak','fish', 'enchilada', 'fajita', 'taco', 'chicken', 'shrimp'])


	def food_worst(food_worst, food_choice_bad):
	    return str(food_choice_bad) == str(food_worst)

	if dish2 == '  ':
	    food_bool2 = [False] * len(df_worst.index)
	elif dish2 != ' ':
	    food_bool_worst = df_worst['cleaan_entity_name'].apply(food_worst, args=(dish2, ))
	    df_worst2 = df_worst[food_bool_worst]
	    st.dataframe(df_worst2)













