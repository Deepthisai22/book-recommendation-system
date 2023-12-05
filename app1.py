#!/usr/bin/env python
# coding: utf-8

# In[16]:


import streamlit as st
import pandas as pd
import numpy as np
import os

# Load your data
file_path_books = os.path.abspath(r'C:\Users\Admin\Project dataset\Books.csv')
books = pd.read_csv(file_path_books, low_memory=False,encoding='latin-1')

file_path_ratings = os.path.abspath(r'C:\Users\Admin\Project dataset\Ratings.csv')
ratings = pd.read_csv(file_path_ratings, encoding='latin-1')

file_path_users = os.path.abspath(r'C:\Users\Admin\Project dataset\Users.csv')
users = pd.read_csv(file_path_users,on_bad_lines='skip',encoding='latin-1')

# Merge ratings with book data
ratings_with_name = ratings.merge(books, on='ISBN')

# Data Preprocessing
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

# Clean the 'Book-Rating' column (Uncomment if needed)
#ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')

# Calculate average rating
ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
   ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

# Calculate user-book similarity
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Calculate similarity scores
from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(pt)


# Define the recommend function
def recommend(book_name):
   # Check if book_name exists in the index
   if book_name not in pt.index:
       return []  # Book not found, return an empty list

   index = np.where(pt.index == book_name)[0][0]
   similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

   data = []
   for i in similar_items:
       item = []
       temp_df = books[books['Book-Title'] == pt.index[i[0]]]
       item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
       item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))

       data.append(item)

   return data


# Create a Streamlit app
st.title("Book Recommendation System")

# Input for the book name
book_name = st.text_input("Enter a book name:")

if st.button("Get Recommendations"):
   if book_name:
       # Call your recommend function with book name
       recommended_books = recommend(book_name)

       # Display the similar book recommendations
       for book in recommended_books:
           st.subheader(book[0])  # Title
           st.write(f"Author: {book[1]}")
   else:
       st.warning("Please enter a book name.")

    
# In[ ]:




