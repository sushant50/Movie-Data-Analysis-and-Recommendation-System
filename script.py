# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from wordcloud import WordCloud, STOPWORDS
import seaborn as sb
import wordcloud
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer

# Reading the datasets using pandas
ratings_ds = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
users_ds = pd.read_csv('users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
movies_ds = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
ds = pd.merge(pd.merge(movies_ds, ratings_ds), users_ds)

#Describing and visualizing the ratings dataset
sb.set_style('whitegrid')
sb.set(font_scale=1.5)
sb.distplot(ratings_ds['rating'].fillna(ratings_ds['rating'].median()))
ratings_ds['rating'].describe()
# Retrieve movies with highest ratings
ds[['title', 'genres', 'rating']].sort_values('rating', ascending=True).head(100)

# Creating a wordcloud visualization containing certain words that feature more often in Movie Titles
movies_ds['title'] = movies_ds['title'].fillna("").astype('str')
docTitle = ' '.join(movies_ds['title'])
wordcloudTitle = WordCloud(stopwords=STOPWORDS, background_color='black', height=1000, width=2000).generate(docTitle)
plot.figure(figsize=(16, 8))
plot.imshow(wordcloudTitle)
plot.axis('off')
plot.show()

# Create a count of genres
genreNames = set()
for name in movies_ds['genres'].str.split('|').values:
    genreNames = genreNames.union(set(name))

# Function that counts the number of times each of the genre keywords appear
def wordCounter(ds, ref_col, count):
    keyCount = dict()
    for s in count:
        keyCount[s] = 0
    for keywordCount in ds[ref_col].str.split('|'):
        if type(keywordCount) == float and pd.isnull(keywordCount):
            continue
        for s in [s for s in keywordCount if s in count]:
            if pd.notnull(s):
                keyCount[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    occurence = []
    for k,v in keyCount.items():
        occurence.append([k,v])
    occurence.sort(key = lambda x:x[1], reverse = True)
    return occurence, keyCount

occurence, dum = wordCounter(movies_ds, 'genres', genreNames)

# Create the dictionary to produce a wordcloud of the movie genres
genres = dict()
trunc_occurences = occurence[0:18]
for name in trunc_occurences:
    genres[name[0]] = name[1]

# Create and display the wordcloud
genre_wordcloud = WordCloud(width=1000,height=400, background_color='white')
genre_wordcloud.generate_from_frequencies(genres)
f, ax = plot.subplots(figsize=(16, 8))
plot.imshow(genre_wordcloud, interpolation="bilinear")
plot.axis('off')
plot.show()

# Break up the big genre string into a string array
movies_ds['genres'] = movies_ds['genres'].str.split('|')
# Convert genres to string value
movies_ds['genres'] = movies_ds['genres'].fillna("").astype('str')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies_ds['genres'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build an array with movie titles
movie_titles = movies_ds['title']
indices = pd.Series(movies_ds.index, index=movies_ds['title'])

# Function to retrieve movies based on the cosine similarity score of movie genres
def genre_recommendations(title):
    if(title is not None):
        idx = indices[title]
        similarityScore = list(enumerate(cosine_sim[idx]))
        similarityScore = sorted(similarityScore, key=lambda x: x[1], reverse=True)
        print(similarityScore)
        similarityScore = similarityScore[1:21]
        print(len(similarityScore))
        movie_indices = [i[0] for i in similarityScore]
        print(movie_indices)
        return movie_titles.iloc[movie_indices]
    else:
        print('here',title)
        return indices.head(20)

# Fill NaN values in user_id and movie_id column with 0 and in rating column with average of all values
ratings_ds['user_id'] = ratings_ds['user_id'].fillna(0)
ratings_ds['movie_id'] = ratings_ds['movie_id'].fillna(0)
ratings_ds['rating'] = ratings_ds['rating'].fillna(ratings_ds['rating'].mean())

# Randomly sample of the ratings dataset
sampleDS = ratings_ds.sample(frac=0.02)

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(sampleDS, test_size=0.2)

# Create two user-item matrices, one for training and another for testing
training_matrix = train_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
test_matrix = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])

from sklearn.metrics.pairwise import pairwise_distances

# Create User Similarity Matrix
user_pcorr = 1 - pairwise_distances(train_data, metric='correlation')
user_pcorr[np.isnan(user_pcorr)] = 0

# Create Item Similarity Matrix
item_pcorr = 1 - pairwise_distances(training_matrix.T, metric='correlation')
item_pcorr[np.isnan(item_pcorr)] = 0

# Function to predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# Function to calculate RMSE
def calculateRMSE(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    rmse = sqrt(mean_squared_error(pred, actual))
    return rmse

# Predict ratings on the training data with both similarity score
user_based_pred = predict(training_matrix, user_pcorr, type='user')
item_based_pred = predict(training_matrix, item_pcorr, type='item')

# Root Mean Square Error on the train data
print('User-based Collaborating Filtering RMSE:')
print(str(calculateRMSE(user_based_pred, training_matrix)))
print('Item-based Collaborating Filtering RMSE:')
print(str(calculateRMSE(item_based_pred, training_matrix)))
# Root Mean Square Error on the test data
print('User-based Collaborating Filtering RMSE:')
print(str(calculateRMSE(user_based_pred, test_matrix)))
print('Item-based Collaborating Filtering RMSE:')
print(str(calculateRMSE(item_based_pred, test_matrix)))
