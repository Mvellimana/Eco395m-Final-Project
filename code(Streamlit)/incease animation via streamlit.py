import ast
import json
import numpy as np
import pandas as pd
import streamlit as ss
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from annotated_text import annotated_text


with ss.sidebar:
	ss.image("https://s3-media2.fl.yelpcdn.com/bphoto/HHyTr44O1_Xp8sYuTpKU8g/o.jpg")
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

	if ss.button("See Our Food Word Cloud"):
		ss.image("wordcloud_food.png")

	if ss.button("See Our Reviews Word Cloud"):
		ss.image("wordcloud_reviews.png")

if choice=="Sentiment Analysis":
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

	df=pd.read_csv("SentimentScore_by_Rest.csv")
	score=ss.slider("All the restaurants above the sentiment score you select",min_value=-0.30,max_value=0.50,value=0.00,step=0.001)
	df_made=df.loc[df["Polarity"]>=score]
	ss.dataframe(df_made)
	ss.image("https://s3-media1.fl.yelpcdn.com/bphoto/Fg2LTmPtlDLo3eCFw_V6Cw/o.jpg",width=520)

	
if choice=="Choose Your Restaurants":	
	ss.image("https://s3-media4.fl.yelpcdn.com/bphoto/XkJSex0R3IgwnO8i4UG2AQ/o.jpg",width=200)

	attribute_choice=ss.multiselect("Tell us your choice",["Takes Reservations", "Offers Delivery","Offers Takeout", "Masks required"])
    
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

	ss.dataframe(df[mask])
	
	
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

	ss.dataframe(df_limit_cols[bool_select])



if choice=="Food Map":
	ss.balloons()





