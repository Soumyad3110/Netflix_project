# This is a simple movie recommender system using Streamlit.
# Import necessary libraries
import pickle
import  pandas as pd
import streamlit as st

# Streamlit Movie Recommender System
# This code is a simple movie recommender system using Streamlit.
# The system uses a pre-trained model to recommend movies based on user input.
# The user selects a movie from a dropdown list, and when they click the "Recommend" button,
# the system returns a list of recommended movies.
# The system uses the cosine similarity metric to find similar movies based on their features.
# The movies and their similarity data are loaded from pickle files.
# The recommender system is built using the Streamlit library, which allows for easy web app development in Python.

# Load the movie data
movies = pickle.load(open('movies.pkl', mode = 'rb'))
print(movies.head())

# Display the first few rows of the movies DataFrame
data = pd.DataFrame(movies)
print(data.head())

# Load the similarity data
similarity = pickle.load(open('similarity.pkl', mode = 'rb'))
print(similarity)

# Define the recommend function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index] # Get the similarity scores for the selected movie
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6] # Get the top 5 similar movies
    recommended_movies = [] # Initialize an empty list to store recommended movies
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title) # Append the movie title to the list
    return recommended_movies
st.title('Movie Recommender System') # Set the title of the app
selected_movie = st.selectbox('Select a movie', movies['title'].values) # Create a dropdown list of movies
if st.button('Recommend'): # Create a button to get recommendations
    recommended_movies = recommend(selected_movie) # Get recommendations
    for i in recommended_movies:
        st.write(i) # Display the recommended movies
                                                                                    
# Add styling and layout
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a footer
st.markdown('---')
st.markdown('<p class="big-font">Made with ❤️ by Movie Recommender</p>', unsafe_allow_html=True)

# Add sidebar information
with st.sidebar:
    st.title('About')
    st.info('This is a movie recommender system that suggests similar movies based on your selection.')
    st.markdown('### How to use:')
    st.write('1. Select a movie from the dropdown')
    st.write('2. Click the Recommend button')
    st.write('3. Get 5 similar movie recommendations')
