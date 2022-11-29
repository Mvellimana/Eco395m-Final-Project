import os
import csv
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.sql import text

load_dotenv()

DATABASE_USERNAME = os.environ["DATABASE_USERNAME"]
DATABASE_PASSWORD = os.environ["DATABASE_PASSWORD"]
DATABASE_HOST = os.environ["DATABASE_HOST"]
DATABASE_PORT = os.environ["DATABASE_PORT"]
DATABASE_DATABASE = os.environ["DATABASE_DATABASE"]

SQLALCHEMY_DATABASE_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_DATABASE}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

sql = '''
    SELECT * FROM business;
'''
with engine.connect().execution_options(autocommit=True) as conn:
    query = conn.execute(text(sql))         
df = pd.DataFrame(query.fetchall())



restaurant_attributes = []

for i in range(len(df)):
    response = requests.get(str(df["business_info"][i]["url"]))
    soup = BeautifulSoup(response.content, "lxml")

    attributes = []
    for attribute in soup.find("yelp-react-root").find("main").find_all("span", {"class": "css-1p9ibgf", "data-font-weight": "semibold"}):
        attributes.append(attribute.text)

    for ii in range(len(attributes)):
        x = {
            "restaurant_name" : df["business_info"][i]["name"],
            "id" : df["business_info"][i]["id"],
            "attributes" : attributes[ii]
        }
        restaurant_attributes.append(x)

    with open(os.path.join("artifacts", "yelp_attributes.csv"), 'w', encoding = "utf-8", newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=["restaurant_name", 'id', 'attributes'])
        dict_writer.writeheader()
        dict_writer.writerows(restaurant_attributes)