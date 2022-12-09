import json

from create_tables import create_tables
from crud import insert_one_business, read_one_business
from yelp_requests import business_search

if __name__ == "__main__":

    LOCATION = "Austin, TX"
    CATEGORIES = "mexican"

    create_tables()

    businesses = business_search(location=LOCATION, categories=CATEGORIES)

    for business in businesses:

        business_id = business["id"]
        business_info_json = json.dumps(business)

        business_dict = {"id": business_id, "business_info": business_info_json}

        if read_one_business(business_id):
            print(f"Business with id={business_id} already exists! Skipping!")
        else:
            print(f"Inserting business with id={business_id} ")
            insert_one_business(business_dict)
