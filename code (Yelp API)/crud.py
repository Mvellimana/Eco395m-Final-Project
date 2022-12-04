from database import engine


def delete_one_business(business_id):
    """Takes an id and deletes the corresponding business."""

    query_template = """
    delete
    from
        business
    where
        id = %(id)s
    """

    with engine.connect() as connection:
        connection.exec_driver_sql(query_template, {"id": business_id})


def read_one_business(business_id):
    """Takes an id and returns a dictionary representing the business,
    if the business exists in the table,
    otherwise returns None."""

    query_template = """
    select
        id,
        business_info
    from
        business
    where
        id = %(id)s
    limit 1
    """

    with engine.connect() as connection:
        result = connection.exec_driver_sql(query_template, {"id": business_id})

        columns = result.keys()
        row = result.first()

        if row is None:
            return None

        return dict(zip(columns, row))


def insert_one_business(business):
    """Takes a dict with keys id and business_info and inserts the business."""

    insert_template = """
        insert into business (id, business_info)
        values (%(id)s, %(business_info)s);
    """

    with engine.connect() as connection:
        connection.exec_driver_sql(insert_template, business)

def insert_one_non_review_business(business):
    """Takes a dict with keys id"""

    insert_template = """
        insert into businesses_with_no_review (id)
        values (%(id)s);
    """

    with engine.connect() as connection:
        connection.exec_driver_sql(insert_template, business)

def insert_one_review(review):
    """Takes a dict with keys id, business_id and review_info and inserts the review."""

    insert_template = """
        insert into review (id, business_id, review_info)
        values (%(id)s, %(business_id)s, %(business_info)s);
    """

    with engine.connect() as connection:
        try:
            connection.exec_driver_sql(insert_template, review)
        except:
            print("something went wrong")
        finally:
            print("the try except is finished")

def pop_business_id_without_reviews():
    query = """
    select
        b.id as business_id
    from
        business b
    left join
        businesses_with_no_review n
    on
        b.id = n.id
    left join
        review r
    on
        b.id = r.business_id
    where
        n.id is Null
        and
        r.id is Null
    limit 1
    """

    with engine.connect() as connection:
        result = connection.exec_driver_sql(query)

        row = result.first()

        if row is None:
            return None

        business_id = row[0]

        return business_id


if __name__ == "__main__":

    import json

    from create_tables import create_tables

    create_tables()

    EXAMPLE_BUSINESS_INFO_DICT = {
        "id": "test-JmL6gV8o-hDzjRLG2TqM4g",
        "alias": "cafe-java-austin-austin-4",
        "name": "Cafe Java - Austin",
        "image_url": "https://s3-media4.fl.yelpcdn.com/bphoto/YdVFCH3MVyXyEUkh7T2zdA/o.jpg",
        "is_closed": False,
        "url": "https://www.yelp.com/biz/cafe-java-austin-austin-4?adjust_creative=y6Exz6DqgD7q0qMRu8zN6w&utm_campaign=yelp_api_v3&utm_medium=api_v3_business_search&utm_source=y6Exz6DqgD7q0qMRu8zN6w",
        "review_count": 1362,
        "categories": [
            {"alias": "coffee", "title": "Coffee & Tea"},
            {"alias": "breakfast_brunch", "title": "Breakfast & Brunch"},
            {"alias": "diners", "title": "Diners"},
        ],
        "rating": 4.5,
        "coordinates": {"latitude": 30.4000609323005, "longitude": -97.7041617306885},
        "transactions": ["delivery"],
        "price": "$$",
        "location": {
            "address1": "11900 Metric Blvd",
            "address2": "Ste K",
            "address3": "",
            "city": "Austin",
            "zip_code": "78758",
            "country": "US",
            "state": "TX",
            "display_address": ["11900 Metric Blvd", "Ste K", "Austin, TX 78758"],
        },
        "phone": "+15123397677",
        "display_phone": "(512) 339-7677",
        "distance": 11737.760957265777,
    }
    EXAMPLE_BUSINESS_INFO = json.dumps(EXAMPLE_BUSINESS_INFO_DICT)
    EXAMPLE_BUSINESS_ID = "test-JmL6gV8o-hDzjRLG2TqM4g"
    EXAMPLE_BUSINESS = {
        "id": EXAMPLE_BUSINESS_ID,
        "business_info": EXAMPLE_BUSINESS_INFO,
    }

    # test that we can insert a business and pull it back out,
    # after testing, clean up by deleting the business if it exists
    try:
        insert_one_business(EXAMPLE_BUSINESS)
        business = read_one_business(EXAMPLE_BUSINESS_ID)
        assert set(business.keys()) == {"id", "business_info"}
        assert business["id"] == EXAMPLE_BUSINESS_ID
        assert business["business_info"] == EXAMPLE_BUSINESS_INFO_DICT
    except Exception as e:
        raise e
    finally:
        pass
        delete_one_business(EXAMPLE_BUSINESS_ID)

    # test that querying a business that doesn't exists returns None
    business = read_one_business("NOT AN ID")
    assert business is None

    pop_business_id_without_reviews()