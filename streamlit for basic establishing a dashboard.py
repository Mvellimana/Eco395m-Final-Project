import streamlit as ss
import pandas as pd
import numpy as np
import altair as alt
import ast


with ss.sidebar:
	ss.image("https://s3-media2.fl.yelpcdn.com/bphoto/HHyTr44O1_Xp8sYuTpKU8g/o.jpg")
	choice=ss.radio("Make your choice",["Main Page","Mexican","Asian","Italian","Food Map"])
ss.sidebar.info("select the one you want!")


if choice=="Main Page":	

	attribute_choice=ss.multiselect("Tell us your choice",["Takes Reservations", "Offers Delivery","Offers Takeout", "Masks required"])
    

	file_name = "mexican_restaurant_attributes.csv"


	def contain_selected_attributes(attributes, attribute_choice_set):
		"""Check if selected attributes are contained, if so, return True, otherwise, return False."""
		def convert_string_to_list(attributes):
			"""Helper function to convert attributes from string representation to list"""
			return ast.literal_eval(attributes)

		attributes_list = convert_string_to_list(attributes)
		return attribute_choice_set.issubset(set(attributes_list))

	df = pd.read_csv(file_name)
	# # adjust selected_attributes to select your attributes
	# selected_attributes = =attribute_choice

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
	

if choice=="Mexican":

	ss.title("Enjoy your Mexican food!")
	ss.image("https://s3-media4.fl.yelpcdn.com/bphoto/uQqWpLnwOdQwtBvJe26pmA/o.jpg")
	ss.subheader("Most frequent words concerned by others! /n Just for your reference")
	ss.image("1.jpeg",width=700) 
   

if choice=="Asian":

	ss.title("Main page")
	ss.image("https://s3-media1.fl.yelpcdn.com/bphoto/Z_O8B_wYYjg_InNABL_12Q/o.jpg")
	
	ss.subheader("Most frequent words concerned by others! /n Just for your reference")
	ss.image("1.jpeg",width=650) #show the words cloud image
	ss.balloons()


if choice=="Italian":

	ss.title("Main page")
	ss.image("https://s3-media2.fl.yelpcdn.com/bphoto/Y5hwoJvM2tLAK9BXb_mcPA/o.jpg")

	ss.subheader("Most frequent words concerned by others! /n Just for your reference")
	ss.image("1.jpeg",width=650) #show the words cloud image

if choice=="Food Map":
	ss.title("Map for all our restaurants")
	df1 = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
    ss.map(df1)


