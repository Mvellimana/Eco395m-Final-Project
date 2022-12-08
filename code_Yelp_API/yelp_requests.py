from nis import cat
import os
from re import L

from dotenv import load_dotenv
import requests

load_dotenv()

YELP_API_KEY = os.environ["YELP_API_KEY"]
HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"}


def business_search_page(location, categories, page):
    """
    Gets a page as a list of up to 50 businesses as dictionaries.
    Pagination starts at 0.
    """
    url = "https://api.yelp.com/v3/businesses/search"
    headers = HEADERS
    params = {
        "location": location,
        "categories": categories,
        "limit": 50,
        "offset": page * 50,
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    response_dict = response.json()

    return response_dict["businesses"]


def business_search(location, categories, page_limit=None):
    """
    Gets up to "page_limit" pages worth of businesses as a list of dictionaries.
    """

    if not page_limit:
        page_limit = float("inf")

    page = 0

    all_businesses = []

    while page < page_limit:

        businesses = business_search_page(
            location=location, categories=categories, page=page
        )
        if not businesses:
            break

        all_businesses += businesses
        page += 1

    return all_businesses


def get_reviews(business_id):

    url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
    headers = HEADERS

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_dict = response.json()

    return response_dict["reviews"]