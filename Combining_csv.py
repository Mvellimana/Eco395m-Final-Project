import os
import glob
import pandas as pd
os.chdir("/Users/Mai/Documents/ECO385M/Eco395m-Final-Project-jordan/artifacts")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
yelp_all_reviews = pd.concat([pd.read_csv(f,encoding = 'ISO-8859-1') for f in all_filenames],axis = 0)
#export to csv
yelp_all_reviews.to_csv( "yelp_all_reviews.csv", index=False, encoding='utf-8-sig')
