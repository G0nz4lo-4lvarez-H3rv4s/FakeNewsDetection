#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from datasets import load_dataset


# In[3]:


df1 = pd.read_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/DataSet True:False/Fake.csv')
df1= df1.drop(columns=['subject','date'])
df1['label'] = 0


# In[4]:


df1


# In[5]:


df2 = pd.read_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/DataSet True:False/True.csv')
df2 = df2.drop(columns=['subject', 'date'])
df2['label'] = 1


# In[6]:


df2


# In[7]:


df3 = pd.read_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/news_articles.csv')
df3 = df3.drop(columns=['author', 'published', 'language', 'site_url', 'main_img_url', 'type', 'title_without_stopwords', 'hasImage', 'text_without_stopwords'])
df3.loc[df3['label'] == 'Real', 'label'] = 1
df3.loc[df3['label'] == 'Fake', 'label'] = 0
df3 = df3.dropna()


# In[8]:


df3


# In[9]:


df_final = pd.concat([df1, df2, df3])


# In[10]:


df_final = df_final.drop_duplicates(subset='text', keep='first')
df_final


# In[11]:


df_final.loc[df_final['label'] == 1].count()


# In[12]:


df_final.loc[df_final['label'] == 0].count()


# In[13]:


print('Final Dataset has got a ' + str((21924/(21924 + 18663)*100)) + '% of True values')
print('Final Dataset has got a ' + str((18663/(21924 + 18663)*100)) + '% of False values')


# In[14]:


df_final = df_final.sample(frac=1)
df_final = df_final.reset_index(drop=True)


# In[15]:


df_final


# In[16]:


df_final.to_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/FinalDataset/final_dataset.csv')


# In[17]:


df_train = df_final.iloc[0:24353, 0:3]
df_train = df_train.reset_index(drop=True)


# In[18]:


df_train


# In[19]:


df_evaluation = df_final.iloc[24353:32470, 0:3]
df_evaluation = df_evaluation.reset_index(drop=True)


# In[20]:


df_evaluation


# In[21]:


df_test = df_final.iloc[32470:40587, 0:3]
df_test = df_test.reset_index(drop=True)


# In[22]:


df_test


# In[23]:


df_train.to_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/FinalDataset/train.csv')
df_evaluation.to_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/FinalDataset/evaluation.csv')
df_test.to_csv('/Users/gonzaloalvarezhervas/TFG/Datasets TFG ~ Fake News/FinalDataset/test.csv')

