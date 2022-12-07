import streamlit as ss
import pandas as pd
import numpy as np
import ast


with ss.sidebar:
	ss.image("https://s3-media2.fl.yelpcdn.com/bphoto/HHyTr44O1_Xp8sYuTpKU8g/o.jpg")
	choice=ss.radio("General View of Our Project",["Overview","Word Clouds","Sentiment Analysis","Choose Your Restaurants","Food Map"])
ss.sidebar.info("select the one you want!")

if choice=="Overview":
	pass


if choice=="Word Clouds":
	ss.image("https://s3-media4.fl.yelpcdn.com/bphoto/3ozWv7yYgawEuBjFIxLjAw/o.jpg",width=350)
	if ss.button("See Our Food Word Cloud"):
		ss.image("wordcloud_food.png")

	if ss.button("See Our Reviews Word Cloud"):
		ss.image("wordcloud_reviews.png")

if choice=="Sentiment Analysis":
	ss.image("https://s3-media1.fl.yelpcdn.com/bphoto/Fg2LTmPtlDLo3eCFw_V6Cw/o.jpg",width=350)
	df=pd.read_csv("SentimentScore_by_Rest.csv")
	score=ss.slider("All the restaurants above the sentiment score you select",min_value=-0.30,max_value=0.50,value=0.00,step=0.001)
	df_made=df.loc[df["Polarity"]>=score]
	ss.dataframe(df_made)

	
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
	# rating_choice=b.text_input("Rating Above")
	rating_choice = b.number_input("Rating Above", step=0.5)


	info_file = 'mexican_restaurant_info.csv'

	df_info = pd.read_csv(info_file)

	# get the boolean values for price
	def price_check(price, price_choice):
		return price== price_choice

	if money_choice == ' ':
		price_bool = [True] * len(df_info.index)
	else:
		price_bool = df_info['price'].apply(price_check, args=(money_choice, ))

	# get the boolean values for rating scores
	def rating_check(rating, rating_input):
		return float(rating) >= float(rating_input)

	rating_bool = df_info['rating'].apply(rating_check, args=(rating_choice, ))

	bool_select = [a and b for a, b in zip(price_bool, rating_bool)]

	df_limit_cols = df_info[['name', 'review_count', 'rating', 'price', 'display_phone']]

	ss.dataframe(df_limit_cols[bool_select])



if choice=="Food Map":
	ss.balloons()





