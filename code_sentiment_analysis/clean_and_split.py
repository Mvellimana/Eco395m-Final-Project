import pandas as pd
import numpy as np

#read the csv and reduce the sample size to focus on 10 categories that we manually chose based on frequency
#we focus on food categories of some popular mexican cuisine since we narrowed down to mexican food
review_data_original=pd.read_csv('artifacts/mexican_reviews_cleaned.csv')

list1=['taco','tortilla','chicken','enchilada','fajita','burrito','fish','shrimp','quesadilla','steak',
'tacos','tortillas','chickens','enchiladas','fajitas','burritos','fishs','shrimps','quesadillas','steaks',
'TACO','TORTILLA','CHICKEN','ENCHILADA','FAJITA','BURRITO','FISH','SHRIMP','QUESADILLA','STEAK',
'TACOS','TORTILLAS','CHICKENS','ENCHILADAS','FAJITAS','BURRITOS','FISHS','SHRIMPS','QUESADILLAS','STEAKS'
]

review_data_original['check']=review_data_original.review.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0)
review_data_big = review_data_original[review_data_original['check']==1]

#we split the data into 4 parts in order to evenly distribute the credits usage on GCP NLP 
review_data_big_split = np.array_split(review_data_big, 4)

review_data_part1 = review_data_big_split[0].drop(['Unnamed: 0'], axis=1)
review_data_part2 = review_data_big_split[1].drop(['Unnamed: 0'], axis=1)
review_data_part3 = review_data_big_split[2].drop(['Unnamed: 0'], axis=1)
review_data_part4 = review_data_big_split[3].drop(['Unnamed: 0'], axis=1)