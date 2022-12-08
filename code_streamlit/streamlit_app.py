import ast
import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from annotated_text import annotated_text
import os




with st.sidebar:
	st.image("https://s3-media2.fl.yelpcdn.com/bphoto/HHyTr44O1_Xp8sYuTpKU8g/o.jpg")
	choice=option_menu(
		menu_title="Final Project",
		options=["Overview","Restaurants Map", "Top Mentioned Words","Choose Your Restaurants", "Sentiment Analysis"],
		icons=["house", "map", "book", "list-task", "list"],
		menu_icon="cast",
		default_index=0,
		)


def load_lottiefile(filepath: str):
	'''load lottifile to add animation effects.'''
	with open(filepath, "r") as f:
		return json.load(f)

if choice=="Overview":

	st.title("Opinions Mining of Yelp Reviews")
	
	lottie_coding=load_lottiefile("animation.json")
	st_lottie(
		lottie_coding,
		speed=1,
		reverse=False,
		loop=True,
		quality="low",
		height=None,
		width=None,
		key=None,
	)

	st.header("Introduction")
	st.markdown("This project aims to conduct an opinions mining analysis based on yelp reviews. The projects includes the use of Google Cloud Platform, \
		SQL, scraping, sentiment analysis, and Streamlit. We were able to view the results of the sentiment analysis based on yelp reviews.")


	st.header("Data Source")
	st.subheader("API Scrapping")
	st.markdown("To start this project, we began with using Yelp’s Fusion API system to gather business ID’s as well as a JSON file associated with those IDs.\
	 This JSON file contained information about the restaurant such as the name of the restaurant, number of reviews, ratings, location, etc. \
	 We followed up this by then using the ID to also gather up to three reviews per restaurant (Yelp’s API limit). \
	 Sadly Yelp also limited the maximum number of characters in a review. Overall, this limited the possibilities of what we could do with sentiment analysis. \
	 That’s when a beacon was lit and Super Scraper Jordan came to our rescue. \
	 Using web scraping, we were able to come up with essentially all the reviews for each Mexican restaurant in Austin.")

	st.subheader("GCP Database")
	st.markdown("For our database we used a PostgreSQL 13 server on the Google Cloud Platform.\
		Which in hindsight might have been a little more powerful than we needed but the server was still cheaper than expected.")

	st.header(" Data Exploration")
	st.markdown("Word clouds (food/reviews) In our daily life, it can be easily understood that one user who is considering what to eat for a meal may come up with ideas referring to other users frequent comments.\
 Words Cloud presenting most frequent search keywords by other users provide an intuitive way. Aiming at giving users more references, we offer 2 choices: \
 one by food and one by reviews.")
	st.markdown("Sometimes users tend to make their specific choice before they \
		begin to find a particular interested restaurant, which makes a filtration function according to a user’s concern important. \
		We offer  such a function with a dropoff menu in our streamlit dashboard.There are 4 attributes in total. \
		Users could choose randomly 1-4 of them, then only  restaurants satisfy all attributes chosen by users appear.")

	st.markdown("Sometimes users are price sensitive, also sometimes users show great interest in restaurants above a particular rating. \
		Nobody wants to pick a low rating restaurant with a higher price. In a word, \
		providing price choice as well as rating choice is meaningful to help a user to find a final destination to eat. In our dashboard, \
		we provide this kind of choice combination. Users can freely either make their choices by price filtration or rating filteration, \
		or even combining these two together to filter.")


	st.header("Methdology")
	st.markdown("Online reviews have the power to drive customers to or away from a business, \
		and tell you what customers like and dislike about it. We performed sentiment and entity analysis on restaurant reviews to uncover emotions in online reviews, \
		to detect trends and patterns that may not be evident at first glance. We also tested the accuracy of the sentiment score using \
		TextBlob’s pre-built model and a Hand-Built Naïve Bayes TF IDF (Bag-of-Words) Model. Our approaches are elaborated in further detail in the sections below.")

	st.subheader("Sentiment Analysis")
	st.markdown("Sentiment Analysis provides an effective way to evaluate the sentiment behind text. \
		It typically uses machine learning algorithms and natural language processing (NLP) to estimate whether the data is positive, \
		neutral or negative")
 
st.markdown("Overall Review Sentiment Analysis")




PATH5 = os.path.join("artifacts", "yelp_map_data.csv")

df5 = (
	pd.read_csv(PATH5)
	)

if choice=="Restaurants Map":
	st.header("Locate the Highest Ratings Mexican Restaurants in Austin")
	st.markdown('Choose the minimum rating.')

	rating = st.slider("Rating", 1.0, float(df5["rating"].max()), 4.5, step = 0.5)
	st.map(df5.query("rating >= @rating")[["latitude", "longitude"]])
	


PATH3 = os.path.join("artifacts", "wordcloud_food.png")
PATH2 = os.path.join("artifacts", "wordcloud_reviews.png")




if choice=="Top Mentioned Words":

	st.title("Top Words/Dishes")
	lottie_coding=load_lottiefile("star.json")
	st_lottie(
		lottie_coding,
		speed=1,
		reverse=False,
		loop=True,
		quality="low",
		height=250,
		width=350,
		key=None,
	)

	st.markdown("This analysis is conducted based on scrapped yelp reviews on Mexican restaurants in Austin. Click on the buttons below to explore them. :smile:")
	st.subheader("Top Mentioned Dishes")
	

	if st.button("Top Mentioned Dishes"):
		st.image(PATH3)

	st.subheader("Top Iterated Words Reviews")
	

	if st.button("Top Iterated Words Reviews"):
		st.image(PATH2)

	


PATH4 = os.path.join("artifacts", "sentiment_score_by_rest.csv")

df4 = (
	pd.read_csv(PATH4)
	)




PATH6 = os.path.join("artifacts", "mexican_restaurant_attributes.csv")
PATH7 = os.path.join("artifacts", "mexican_restaurant_info.csv")



if choice=="Choose Your Restaurants":
	st.title("Choose Your Restaurants")

	st.image("https://s3-media4.fl.yelpcdn.com/bphoto/XkJSex0R3IgwnO8i4UG2AQ/o.jpg",width=200)
	st.subheader("Select your Favorite Restaurants Based on Attributes")

	attribute_choice=st.multiselect("Filter by attributes",["Takes Reservations", "Offers Delivery","Offers Takeout", "Masks required"])
    
	
	df6 = pd.read_csv(PATH6)

	def contain_selected_attributes(attributes, attribute_choice_set):
		"""Check if selected attributes are contained, if so, return True, otherwise, return False."""
		def convert_string_to_list(attributes):
			"""Helper function to convert attributes from string representation to list"""
			return ast.literal_eval(attributes)

		attributes_list = convert_string_to_list(attributes)
		return attribute_choice_set.issubset(set(attributes_list))


	mask = df6["attributes"].apply(
	    contain_selected_attributes, args=(set(attribute_choice),)
	)

	st.dataframe(df6[mask])
	
	
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


PATH8 = os.path.join("artifacts", "best_rest.csv")
PATH9 = os.path.join("artifacts", "worst_rest.csv")	


if choice=="Sentiment Analysis":

	st.title("Sentiment Analysis")

	lottie_coding=load_lottiefile("avocado.json")
	st_lottie(
		lottie_coding,
		speed=1,
		reverse=False,
		loop=True,
		quality="low",
		height=250,
		width=350,
		key=None,
	)

	st.header("Pre-Built Algorthim vs Hand-Built Algorthim")
	st.markdown("We have conducted 2 different models to assess the sentiment analysis. The first model is a **TextBlob** model which is a *Pre-Built Algorthim* that uses\
		reviews text as input. The other model is a **Naïve Bayes- TF IDF** which is a *Hand-Built Model* that uses rating as input which could be a proxy for reviewers sentiment.\
		The objective is to compare the accuracy of both models.")
	col1, col2 = st.columns(2)
	col1.metric("Pre-Built Algorthim", "89.67% Accuracy", "TextBlob")
	col2.metric("Hand-Built Algorthim", "83.8% Accuracy", "Naïve Bayes- TF IDF")
	st.markdown("So, we are using the **TextBlob** model results to show them below.")
	

	st.header("Find the Sentiment Score for Mexican Restaurants")
	st.markdown("Slide right for restaurants with higher sentiment analysis, and left to include the lower.")
	score=st.slider("Select the minimum sentiment score.",min_value=-1.0,max_value=1.0,value=0.00,step=0.001)
	df4_made=df4.loc[df4["Polarity"]>=score]
	st.dataframe(df4_made)
	
	

	st.header("Find the Best and the worst restaurants from the selectbox")
	st.markdown("Pick a specific dish to find the top 5 and worst 5 restaurants based on Google Cloud NLP.")
	df_best = pd.read_csv(PATH8)
	df_worst = pd.read_csv(PATH9)

	dish = st.selectbox(
	    "Choose dishes you want, Find the Best and Leave the Worst!",
	    [' ','tortilla', 'burrito', 'quesadilla', 'steak','fish', 'enchilada', 'fajita', 'taco', 'chicken', 'shrimp'])
	
	def food_check(food, food_choice):
	    return str(food_choice) == str(food)

	if dish == ' ':
	    food_bool = [False] * len(df_best.index)
	elif dish != ' ':
	    food_bool_best = df_best['cleaan_entity_name'].apply(food_check, args=(dish, ))
	    food_bool_worst = df_best['cleaan_entity_name'].apply(food_check, args=(dish, ))
	    df_1 = df_best[food_bool_best]
	    df_2 = df_worst[food_bool_worst]
	    df_comb = df_1.append(df_2)
	    st.dataframe(df_comb)













