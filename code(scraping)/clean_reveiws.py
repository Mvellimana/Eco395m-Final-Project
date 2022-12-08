import pandas as pd



df1 = pd.read_csv("artifacts/mexican_reviews.csv")

dict = list(df1.T.to_dict().values())


new_dict = []

for i in range(len(dict)):
    clean_review = dict[i]["review"].replace("é", "e").replace("á", "a").replace("í", "i").replace("Ñ", "N").replace("ñ", "n").replace("ó", "o").replace("ú", "u").replace("ü", "u").replace("Ó", "O").replace("Á", "A")

    x = {
        "id" : dict[i]["id"],
        "restaurant_name" : dict[i]["restaurant_name"],
        "number" : dict[i]["number"],
        "review" : clean_review,
        "rating" : dict[i]["rating"],
        "elite_status" : dict[i]["elite_status"],
        "number_of_photos_included" : dict[i]["number_of_photos_included"],
        "review_attributes" : dict[i]["review_attributes"]
        }
    new_dict.append(x)

df2 = pd.DataFrame(new_dict)

df2.to_csv("artifacts/mexican_reviews_cleaned.csv")



