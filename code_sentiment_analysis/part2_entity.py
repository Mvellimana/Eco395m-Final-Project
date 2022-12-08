#review_data_part2['dfs']=review_data_part2.apply(lambda x: get_res(x),axis=1)

review_data_entity_part2=pd.DataFrame()
for i in range(review_data_part2.shape[0]):
    review_data_entity_part2=pd.concat([review_data_entity_part2, review_data_part2.dfs.iloc[i]], axis=0)
# review_data_entity_part2.to_csv('code_Sentiment_Analysis/working_csvs/review_data_entity_part2.csv')

review_data_entity_part2=pd.read_csv('code_Sentiment_Analysis/working_csvs/review_data_entity_part2.csv')

review_data_entity_part2_split = np.array_split(review_data_entity_part2, 8)

review_data_entity_part2_subsection1 = review_data_entity_part2_split[0]
review_data_entity_part2_subsection2 = review_data_entity_part2_split[1]
review_data_entity_part2_subsection3 = review_data_entity_part2_split[2]
review_data_entity_part2_subsection4 = review_data_entity_part2_split[3]
review_data_entity_part2_subsection5 = review_data_entity_part2_split[4]
review_data_entity_part2_subsection6 = review_data_entity_part2_split[5]
review_data_entity_part2_subsection7 = review_data_entity_part2_split[6]
review_data_entity_part2_subsection8 = review_data_entity_part2_split[7]

def get_cleanname(x):
    a=''
    for i in list2:
        if i in x.lower():
            a=i
    return a

list2=['taco','tortilla','chicken','enchilada','fajita','burrito','fish','shrimp','quesadilla','steak']

review_data_entity_part2_subsection1['check']=review_data_entity_part2_subsection1.entity_name.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection1 = review_data_entity_part2_subsection1[review_data_entity_part2_subsection1['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection1['cleaan_entity_name']= review_data_entity_part2_subsection1.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection1 = review_data_entity_part2_subsection1.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection1_grouped=review_data_entity_part2_subsection1.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)


review_data_entity_part2_subsection2['check']=review_data_entity_part2_subsection2.entity_name.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection2 = review_data_entity_part2_subsection2[review_data_entity_part2_subsection2['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection2['cleaan_entity_name']= review_data_entity_part2_subsection2.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection2 = review_data_entity_part2_subsection2.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection2_grouped=review_data_entity_part2_subsection2.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)

review_data_entity_part2_subsection3['check']=review_data_entity_part2_subsection3.entity_name.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection3 = review_data_entity_part2_subsection3[review_data_entity_part2_subsection3['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection3['cleaan_entity_name']= review_data_entity_part2_subsection3.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection3 = review_data_entity_part2_subsection3.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection3_grouped=review_data_entity_part2_subsection3.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)

review_data_entity_part2_subsection4['check']=review_data_entity_part2_subsection4.entity_name.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection4 = review_data_entity_part2_subsection4[review_data_entity_part2_subsection4['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection4['cleaan_entity_name']= review_data_entity_part2_subsection4.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection4 = review_data_entity_part2_subsection4.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection4_grouped=review_data_entity_part2_subsection4.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)

review_data_entity_part2_subsection5['check']=review_data_entity_part2_subsection5.entity_name.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection5 = review_data_entity_part2_subsection5[review_data_entity_part2_subsection5['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection5['cleaan_entity_name']= review_data_entity_part2_subsection5.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection5 = review_data_entity_part2_subsection5.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection5_grouped=review_data_entity_part2_subsection5.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)

review_data_entity_part2_subsection6['check']=review_data_entity_part2_subsection6.entity_name.apply(lambda x: 1 if any(substring in str(x) for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection6 = review_data_entity_part2_subsection6[review_data_entity_part2_subsection6['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection6['cleaan_entity_name']= review_data_entity_part2_subsection6.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection6 = review_data_entity_part2_subsection6.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection6_grouped=review_data_entity_part2_subsection6.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)

review_data_entity_part2_subsection7['check']=review_data_entity_part2_subsection7.entity_name.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection7 = review_data_entity_part2_subsection7[review_data_entity_part2_subsection7['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection7['cleaan_entity_name']= review_data_entity_part2_subsection7.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection7 = review_data_entity_part2_subsection7.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection7_grouped=review_data_entity_part2_subsection7.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)

review_data_entity_part2_subsection8['check']=review_data_entity_part2_subsection8.entity_name.apply(lambda x: 1 if any(substring in x for substring in list1)==True else 0).astype(int)
review_data_entity_part2_subsection8 = review_data_entity_part2_subsection8[review_data_entity_part2_subsection8['check']==1].sort_values(by=['entity_sentiment_score'],ascending=False)
review_data_entity_part2_subsection8['cleaan_entity_name']= review_data_entity_part2_subsection8.entity_name.apply(lambda x: get_cleanname(x))
review_data_entity_part2_subsection8 = review_data_entity_part2_subsection8.groupby(['cleaan_entity_name','restaurant_name'])['entity_sentiment_score'].mean()
review_data_entity_part2_subsection8_grouped=review_data_entity_part2_subsection8.reset_index().sort_values(by=['cleaan_entity_name','entity_sentiment_score'],ascending=False)


frames = [review_data_entity_part2_subsection1_grouped, review_data_entity_part2_subsection2_grouped, review_data_entity_part2_subsection3_grouped, 
          review_data_entity_part2_subsection4_grouped, review_data_entity_part2_subsection5_grouped, review_data_entity_part2_subsection6_grouped, 
          review_data_entity_part2_subsection7_grouped, review_data_entity_part2_subsection8_grouped]

review_data_entity_part2_grouped = pd.concat(frames)

review_data_entity_part2_grouped.to_csv('code_Sentiment_Analysis/working_csvs/review_data_entity_part2_grouped.csv')