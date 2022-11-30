import os
import csv
import requests
import os
import csv
import requests
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


review_plus_rating = []

for j in range(51):
    response = requests.get(str(df["business_info"][2]["url"]) + "&start=" + str(j) + str(0))
    soup = BeautifulSoup(response.content, "lxml")

    reviews = []
    ratings = []

    for i in range(10):
        review = soup.find("yelp-react-root").find_all("p", {"class" : "comment__09f24__gu0rG css-qgunke"})[i].find("span").decode_contents()
        reviews.append(review)

    for ii in range(10):
        rating = soup.find("yelp-react-root").find("main").find_all("div", {"class" : "five-stars__09f24__mBKym five-stars--regular__09f24__DgBNj display--inline-block__09f24__fEDiJ border-color--default__09f24__NPAKY", "role" : "img"})[i]["aria-label"]
        ratings.append(rating)


    for iii in range(10):
        x = {
            "restaurant_name" : "Granny's Tacos",
            "id" : df["business_info"][2]["id"],
            "number" : iii+1+(j*10),
            "review" : reviews[iii],
            "rating" : ratings[iii]
        }
        review_plus_rating.append(x)

        with open(os.path.join("artifacts", "grannys_taco_reviews_plus_rating.csv"), "w", encoding = "utf-8", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=["restaurant_name", "id", "number", "review", "rating"])
            dict_writer.writeheader()
            dict_writer.writerows(review_plus_rating)