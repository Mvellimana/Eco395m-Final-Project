from database import engine


def make_businesses_table():

    cmd = """
        create table if not exists business (
            id varchar primary key,
            business_info JSON
        );
        create table if not exists review (
            id varchar primary key,
            business_id varchar references business(id),
            review_info JSON
        );

        create table if not exists businesses_with_no_review (
            id varchar primary key
        );
    """

    with engine.connect() as connection:
        connection.exec_driver_sql(cmd)


def create_tables():
    make_businesses_table()


if __name__ == "__main__":
    create_tables()