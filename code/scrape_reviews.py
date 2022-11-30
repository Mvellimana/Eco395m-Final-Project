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



dict_reviews = []

for i in range(0, 588):
    for j in range(50):
        response = requests.get(str(df["business_info"][i]["url"]) + "&start="+ str(j) + str(0))
        soup = BeautifulSoup(response.content, "lxml")

        reviews = []
        try:
            for review in soup.find("yelp-react-root").find_all("p", {"class" : "comment__09f24__gu0rG css-qgunke"}):
                reviews.append(review.find("span").decode_contents())
        except (AttributeError, IndexError):
            review = "no review"
            reviews.append(review)


        ratings = []
        for ik in range(0, 10):
            try:
                xx = soup.find("yelp-react-root").find("main").find_all("div", {"class" : "five-stars__09f24__mBKym five-stars--regular__09f24__DgBNj display--inline-block__09f24__fEDiJ border-color--default__09f24__NPAKY", "role" : "img"})[ik]["aria-label"]
                ratings.append(xx)
            except (AttributeError, IndexError):
                xx = "no rating"
                ratings.append(xx)

        elite_status = []
        for ij in range(1, 11):
            try:
                status = soup.find("yelp-react-root").find("main").find_all("div", {"class" : "user-passport-info border-color--default__09f24__NPAKY"})[ij].find("span", {"class" : "css-1adhs7a"}).decode_contents()
                elite_status.append(status)

            except (AttributeError, IndexError):
                status2 = "none"
                elite_status.append(status2)

        number_of_photos = []
        for ip in range(0, 10):
            try:
                num_photo = len(soup.find("yelp-react-root").find("main").find_all("div", {"class" : "review__09f24__oHr9V border-color--default__09f24__NPAKY"})[ip].find_all("div", {"class" : "photo-container-small__09f24__obhgq border-color--default__09f24__NPAKY"})) + len(soup.find("yelp-react-root").find("main").find_all("div", {"class" : "review__09f24__oHr9V border-color--default__09f24__NPAKY"})[ip].find_all("div", {"class" : "photo-container-large__09f24__fUgaj border-color--default__09f24__NPAKY"}))
                number_of_photos.append(num_photo)

            except (AttributeError, IndexError):
                num_photo2 = 0
                number_of_photos.append(num_photo2)

        review_attributes = []
        for ir in range(0, 10):
            try:
                useful = soup.find("yelp-react-root").find("main").find_all("div", {"class" : "review__09f24__oHr9V border-color--default__09f24__NPAKY"})[ir].find_all("span", {"class" : "css-12i50in"})[0].find("span", {"class" : "css-1lr1m88"}).decode_contents().replace("<!-- -->", "")
            except (AttributeError, IndexError):
                useful = 0
            
            try:
                funny = soup.find("yelp-react-root").find("main").find_all("div", {"class" : "review__09f24__oHr9V border-color--default__09f24__NPAKY"})[ir].find_all("span", {"class" : "css-12i50in"})[1].find("span", {"class" : "css-1lr1m88"}).decode_contents().replace("<!-- -->", "")
            except (AttributeError, IndexError):
                funny = 0

            try: 
                cool = soup.find("yelp-react-root").find("main").find_all("div", {"class" : "review__09f24__oHr9V border-color--default__09f24__NPAKY"})[ir].find_all("span", {"class" : "css-12i50in"})[2].find("span", {"class" : "css-1lr1m88"}).decode_contents().replace("<!-- -->", "")
            except (AttributeError, IndexError):
                cool = 0

            dict_attr = {
                "Useful" : useful,
                "Funny" : funny,
                "Cool" : cool
            }

            review_attributes.append(dict_attr)



        for ii in range(len(reviews)):
            x = {
                "id" : df["business_info"][i]["id"],
                "restaurant_name" : df["business_info"][i]["name"],
                "number" : ii+1+(j*10),
                "review" : reviews[ii].replace("<br/>", " ").replace("---", ""),
                "rating" : ratings[ii],
                "elite_status" : elite_status[ii],
                "number_of_photos_included" : number_of_photos[ii],
                "review_attributes" : review_attributes[ii]
            }
            dict_reviews.append(x)



    with open(os.path.join("updated_artifacts", str(df["business_info"][i]["name"]) + ".csv"), "w", encoding = "utf-8", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=["id", "restaurant_name", "number", "review", "rating", "elite_status", "number_of_photos_included", "review_attributes"])
        dict_writer.writeheader()
        dict_writer.writerows(dict_reviews)
