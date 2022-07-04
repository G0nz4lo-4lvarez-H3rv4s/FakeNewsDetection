#!/usr/bin/env python
import pandas as pd

from datasets import load_dataset

df1 = pd.read_csv('../Fake.csv')
df1= df1.drop(columns=['subject','date'])
df1['label'] = 0

df2 = pd.read_csv('../True.csv')
df2 = df2.drop(columns=['subject', 'date'])
df2['label'] = 1

df3 = pd.read_csv('../news_articles.csv')
df3 = df3.drop(columns=['author', 'published', 'language', 'site_url', 'main_img_url', 'type', 'title_without_stopwords', 'hasImage', 'text_without_stopwords'])
df3.loc[df3['label'] == 'Real', 'label'] = 1
df3.loc[df3['label'] == 'Fake', 'label'] = 0
df3 = df3.dropna()

df_final = pd.concat([df1, df2, df3])


df_final = df_final.drop_duplicates(subset='text', keep='first')
df_final.loc[df_final['label'] == 1].count()
df_final.loc[df_final['label'] == 0].count()

print('Final Dataset has got a ' + str((21924/(21924 + 18663)*100)) + '% of True values')
print('Final Dataset has got a ' + str((18663/(21924 + 18663)*100)) + '% of False values')

df_final = df_final.sample(frac=1)
df_final = df_final.reset_index(drop=True)

df_final.to_csv('../final_dataset.csv')


df_train = df_final.iloc[0:24353, 0:3]
df_train = df_train.reset_index(drop=True)

df_evaluation = df_final.iloc[24353:32470, 0:3]
df_evaluation = df_evaluation.reset_index(drop=True)

df_test = df_final.iloc[32470:40587, 0:3]
df_test = df_test.reset_index(drop=True)

df_train.to_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/FinalDataset/train.csv')
df_evaluation.to_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/FinalDataset/evaluation.csv')
df_test.to_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/FinalDataset/test.csv')
