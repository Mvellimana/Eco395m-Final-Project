import ast
import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from annotated_text import annotated_text
import os



PATH5 = os.path.join("artifacts", "yelp_map_data.csv")

df5 = (
	pd.read_csv(PATH5)
	)





with st.sidebar:
	st.image("https://s3-media2.fl.yelpcdn.com/bphoto/HHyTr44O1_Xp8sYuTpKU8g/o.jpg")
	choice=option_menu(
		menu_title="Final Project",
		options=["Overview","Word Clouds","Sentiment Analysis","Choose Your Restaurants","Food Map"],
		icons=["house","book","envelope","book","envelope"],
		menu_icon="cast",
		default_index=0,
		)


def load_lottiefile(filepath: str):
	'''load lottifile to add animation effects.'''
	with open(filepath, "r") as f:
		return json.load(f)

if choice=="Overview":
	
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


if choice=="Word Clouds":
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

	if st.button("Top Mentioned Dishes"):
		st.image("wordcloud_food.png")

	if st.button("Top Iterated Words Reviews"):
		st.image("wordcloud_reviews.png")


PATH4 = os.path.join("artifacts", "sentiment_score_by_rest.csv")

df4 = (
	pd.read_csv(PATH4)
	)


if choice=="Sentiment Analysis":

	st.title("this is title")
	st.header("this is header")
	st.markdown("this is text to explain this part shortly")
	col1, col2 = st.columns(2)
	col1.metric("Pre-Built Algorthim", "89.67% Accuracy", "TextBlob")
	col2.metric("Hand-Built Algorthim", "83.8% Accuracy", "Naive Bayes- TF IDF")
	

	st.header("this is header here")
	st.subheader("Slide to your Favorite")
	score=st.slider("All the restaurants above the sentiment score you select",min_value=-1.0,max_value=1.0,value=0.00,step=0.001)
	df4_made=df4.loc[df4["Polarity"]>=score]
	st.dataframe(df4_made)
	st.image("https://s3-media1.fl.yelpcdn.com/bphoto/Fg2LTmPtlDLo3eCFw_V6Cw/o.jpg",width=520)


	
if choice=="Choose Your Restaurants":	
	st.image("https://s3-media4.fl.yelpcdn.com/bphoto/XkJSex0R3IgwnO8i4UG2AQ/o.jpg",width=200)

	attribute_choice=st.multiselect("Tell us your choice",["Takes Reservations", "Offers Delivery","Offers Takeout", "Masks required"])
    
	file_name = "mexican_restaurant_attributes.csv"
	df = pd.read_csv(file_name)

	def contain_selected_attributes(attributes, attribute_choice_set):
		"""Check if selected attributes are contained, if so, return True, otherwise, return False."""
		def convert_string_to_list(attributes):
			"""Helper function to convert attributes from string representation to list"""
			return ast.literal_eval(attributes)

		attributes_list = convert_string_to_list(attributes)
		return attribute_choice_set.issubset(set(attributes_list))


	mask = df["attributes"].apply(
	    contain_selected_attributes, args=(set(attribute_choice),)
	)

	st.dataframe(df[mask])
	
	
	a,b=ss.columns([3,1])
	money_choice=a.selectbox("Price Range",[" ","$$$","$$","$"]) 
	rating_choice = b.number_input("Rating Above", step=0.5)


	info_file = 'mexican_restaurant_info.csv'

	df_info = pd.read_csv(info_file)

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





if choice=="Food Map":
	st.header("Choose the Minimum Rating for your Favorite Mexican Restaurants")
	rating = st.slider("Rating", 1.0, float(df5["rating"].max()), 4.5, step = 0.5)
	st.map(df5.query("rating >= @rating")[["latitude", "longitude"]])
	st.markdown('jkdfkjf')





