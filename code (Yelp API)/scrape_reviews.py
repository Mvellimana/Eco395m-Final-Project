import json

from create_tables import create_tables
from crud import pop_business_id_without_reviews, insert_one_review
from yelp_requests import get_reviews

if __name__ == "__main__":

    create_tables()

    while True:

        business_id = pop_business_id_without_reviews()

        if not business_id:
            print("All businesses have associated reviews.")
            break

        reviews = get_reviews(business_id)

        for review in reviews:

            review_id = review["id"]
            review_info_json = json.dumps(review)

            review_dict = {
                "id": review_id,
                "business_id": business_id,
                "business_info": review_info_json,
            }

            print(
                f"Inserting review for business_id={business_id} with review_id={review_id}"
            )
            insert_one_review(review_dict)