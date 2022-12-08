frms = [review_data_entity_part1_grouped, review_data_entity_part2_grouped, review_data_entity_part3_grouped, review_data_entity_part4_grouped]

review_data_entity_grouped = pd.concat(frms)

review_data_entity_grouped_final=review_data_entity_grouped[['cleaan_entity_name', 'restaurant_name',
       'entity_sentiment_score']]restaurant_namecleaan_entity_name
df=review_data_entity_grouped_final.groupby(['cleaan_entity_name','restaurant_name']).entity_sentiment_score.mean().reset_index()

df1=df.groupby(['cleaan_entity_name'])['entity_sentiment_score'].nlargest(5).reset_index()
df2=df.reset_index()
df3=df1.merge(df2,left_on='level_1',right_on='index')
df4=df3[['cleaan_entity_name_x','restaurant_name','entity_sentiment_score_x']]
df4.columns=['cleaan_entity_name','restaurant_name','entity_sentiment_score']
df4.to_csv('artifacts/best_rest.csv')

df1=df.groupby(['cleaan_entity_name'])['entity_sentiment_score'].nsmallest(5).reset_index()
df2=df.reset_index()
df3=df1.merge(df2,left_on='level_1',right_on='index')
df4=df3[['cleaan_entity_name_x','restaurant_name','entity_sentiment_score_x']]
df4.columns=['cleaan_entity_name','restaurant_name','entity_sentiment_score']
df4.to_csv('artifacts/worst_rest.csv')
