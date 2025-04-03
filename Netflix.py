import re # for regex operations
import ast # for converting string to list
import pickle # for saving and loading data, dumped as a binary file

import pandas as pd # for data manipulation and analysis
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization


import sys # for system-specific parameters and functions
import os # for operating system dependent functionality

import nltk # for natural language processing
from nltk.corpus import stopwords # for accessing NLTK's corpus
from nltk.stem import PorterStemmer # for stemming words
from sklearn.feature_extraction.text import CountVectorizer # for converting text to a matrix of token counts
from sklearn.metrics.pairwise import cosine_similarity # for calculating cosine similarity
from sklearn.cluster import DBSCAN # for clustering 

import warnings # for handling warnings
warnings.filterwarnings('ignore') # ignore warnings

movies = pd.read_csv("C:/Users/KIIT/Downloads/tmdb_5000_movies.csv") # load movies dataset
credits = pd.read_csv("C:/Users/KIIT/Downloads/tmdb_5000_credits.csv") # load credits dataset

print(movies.head(2)) # print first 2 rows of movies dataset
print(credits.head(2)) # print first 2 rows of credits dataset

print(movies.columns) # print columns of movies dataset
print(credits.columns) # print columns of credits dataset

print(movies.shape) # print shape of movies dataset
print(credits.shape) # print shape of credits dataset

print(movies.info()) # print info of movies dataset
print(credits.info()) # print info of credits dataset

print(movies.isnull().sum()) # print null values in movies dataset
print(credits.isnull().sum()) # print null values in credits dataset

print(movies.duplicated().sum()) # print duplicated values in movies dataset
print(credits.duplicated().sum()) # print duplicated values in credits dataset

# Merge the dataframees on the basis of title

movies = movies.merge(credits, on='title') # merge movies and credits dataset on title column
print(movies.shape) # print shape of merged dataset

# Choose the relevant features for the recommendation system    

# 1. movie_id
# 2. title
# 3. overview
# 4. genres
# 5. keywords
# 6. cast
# 7. crew

# Drop the unnecessary columns from the dataset

df = movies[['movie_id', 'title', 'overview','genres', 'keywords', 'cast', 'crew']]
print(df.head(2)) # print first 2 rows of relevant features                                                                                                                                                 
print(df.shape) # print shape of relevant features
print(df.info()) # print info of relevant features  

print(df.isnull().sum()) # print null values in relevant features
print(df.duplicated().sum()) # print duplicated values in relevant features

print(df['overview'].isnull().sum()) # print null values in overview column

# Fill null values in overview with empty string
df['overview'] = df['overview'].fillna('')

# Convert genres from string to list and extract names
def convert_genres(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Convert keywords from string to list and extract names
def convert_keywords(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Get top 3 cast members
def fetch_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

# Get director name from crew
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Apply conversions
df['genres'] = df['genres'].apply(convert_genres)
df['keywords'] = df['keywords'].apply(convert_keywords)
df['cast'] = df['cast'].apply(fetch_cast)
df['crew'] = df['crew'].apply(fetch_director)

# Remove spaces from names
df['genres'] = df['genres'].apply(lambda x: [i.replace(" ","") for i in x])
df['keywords'] = df['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
df['cast'] = df['cast'].apply(lambda x: [i.replace(" ","") for i in x])
df['crew'] = df['crew'].apply(lambda x: [i.replace(" ","") for i in x])

# Create tags by combining features
df['tags'] = df['overview'] + ' ' + df['genres'].apply(lambda x: " ".join(x)) + ' ' + df['keywords'].apply(lambda x: " ".join(x)) + ' ' + df['cast'].apply(lambda x: " ".join(x)) + ' ' + df['crew'].apply(lambda x: " ".join(x))

# Final dataframe
df = df[['movie_id', 'title', 'overview', 'tags']]
print(df.head(2)) # print first 2 rows of final dataframe
print(df.shape) # print shape of final dataframe
print(df.info()) # print info of final dataframe

# Convert tags to lowercase
df['tags'] = df['tags'].apply(lambda x: x.lower())

# Initialize NLTK components
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Text preprocessing function
def preprocess_text(text):
    # Tokenization and cleaning
    tokens = re.split('\W+', text)
    
    # Remove stopwords and apply stemming
    tokens = [ps.stem(word) for word in tokens if word.lower() not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing to tags
df['tags'] = df['tags'].apply(preprocess_text)

# Encode movie IDs to numerical indices
movie_indices = pd.Series(df.index, index=df['title'])

# Create count matrix and calculate cosine similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)
print(similarity) # print similarity matrix
# Save similarity matrix
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Load similarity matrix    
def recommend_movies(movie, n=5):
    # Find the index of the movie
    movie_index = df[df['title'] == movie].index[0]
    
    # Get similarity scores for the movie
    distances = similarity[movie_index]
    
    # Get list of similar movie indices (sorted)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]
    
    # Print recommended movies
    print(f"\nRecommended movies similar to {movie}:")
    for i, score in movie_list:
        print(f"{df['title'].iloc[i]} (similarity score: {score:.2f})")


# Load the similarity matrix from the file
similarity = pickle.load(open('similarity.pkl', 'rb'))


# Example usage
movie_name = input("Enter a movie name: ")
recommend_movies(movie_name)