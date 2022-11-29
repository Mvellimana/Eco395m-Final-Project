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



all_restaurant_info = []

for i in range(len(df)):
    x = df["business_info"][i]
    all_restaurant_info.append(x)


with open(os.path.join("artifacts", "restaurant_info.csv"), "w", encoding = "utf-8", newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=["id", "alias", "name", "image_url", "is_closed", "url", "review_count", "categories", "rating", "coordinates", "transactions", "price", "location", "phone", "display_phone", "distance"])
    dict_writer.writeheader()
    dict_writer.writerows(all_restaurant_info)

