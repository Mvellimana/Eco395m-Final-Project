import json

from create_tables import create_tables
from crud import pop_business_id_without_reviews, insert_one_review
from yelp_requests import get_reviews

if __name__ == "__main__":

    create_tables()

    while True:
        print("checking for business reviews")
        business_id = pop_business_id_without_reviews()
        print("Checked business reviews")
        
        if not business_id:
            print("All businesses have associated reviews.")
            break
        
        print("getting reviews for business id:" + str(business_id))
        reviews = get_reviews(business_id)
        if len(reviews) == 0:
            print("no reviews found for " + str(business_id))
            insert_one_non_review_business(business_id)
            continue

        print("num of reviews = " + str(len(reviews)))
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