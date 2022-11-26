import os
import json
import csv
import requests
from bs4 import BeautifulSoup
import re

reviews = []
for i in range(26):
    response = requests.get("https://www.yelp.com/biz/vamonos-austin?start=" + str(i) + str(0))
    soup = BeautifulSoup(response.content, "lxml")
    for review in soup.find("yelp-react-root").find_all("p", {"class" : "comment__09f24__gu0rG css-qgunke"}):
        reviews.append(review.find("span").decode_contents())

dict_reviews = []
for i in range(len(reviews)):
    x = {
        "number" : i+1,
        "review" : reviews[i]
    }
    dict_reviews.append(x)

with open("yelp_reviews.csv", 'w', encoding = "utf-8", newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=['number', 'review'])
    dict_writer.writeheader()
    dict_writer.writerows(dict_reviews)

