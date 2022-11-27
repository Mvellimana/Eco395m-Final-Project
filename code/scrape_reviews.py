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



dict_reviews = []

for i in range(587):
    for j in range(25):
        response = requests.get(str(df["business_info"][i+406]["url"]) + "&start="+ str(j) + str(0))
        soup = BeautifulSoup(response.content, "lxml")

        reviews = []
        for review in soup.find("yelp-react-root").find_all("p", {"class" : "comment__09f24__gu0rG css-qgunke"}):
            reviews.append(review.find("span").decode_contents())

        for ii in range(len(reviews)):
            x = {
                "restaurant_name" : df["business_info"][i+406]["name"],
                "number" : ii+1+(j*10),
                "review" : re.sub('[^a-zA-Z0-9-_.]', ' ', reviews[ii].replace("<br/>", " ").replace("---", ""))
            }
            dict_reviews.append(x)


    with open(os.path.join("artifacts", "yelp_all_reviews_" + df["business_info"][i+406]["name"] + ".csv"), 'w', encoding = "utf-8", newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=["restaurant_name", 'number', 'review'])
        dict_writer.writeheader()
        dict_writer.writerows(dict_reviews)
