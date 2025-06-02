# 출처 - https://ysg2997.tistory.com/27

import pandas as pd
import networkx as nx
from io import BytesIO
from zipfile import ZipFile
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from urllib.request import urlopen
from collections import defaultdict

# Download and extract the MovieLens 100k dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
with urlopen(url) as zurl:
    with ZipFile(BytesIO(zurl.read())) as zfile:
        zfile.extractall('.')

# Load ratings and movie titles
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(2), names=['movie_id', 'title'], encoding='latin-1')

# Use only ratings of 4 or higher
ratings = ratings[ratings.rating >= 4]

# Initialize a dictionary to count how often each pair of movies is liked together
# Uses defaultdict to automatically assign 0 to any new key
pairs = defaultdict(int)

# Loop through the entire list of users
for group in ratings.groupby("user_id"):
    # List of movie IDs rated by the current user
    user_movies = list(group[1]["movie_id"])

    # Count every time two movies are liked together
    for i in range(len(user_movies)):
        for j in range(i + 1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])] += 1

# Create a networkx graph
G = nx.Graph()

# Create an edge between movies that are liked together
for pair in pairs:
    movie1, movie2 = pair
    score = pairs[pair]

    # Only create the edge if the score is 20 or higher
    if score >= 20:
        G.add_edge(movie1, movie2, weight=score)

# Print the total number of nodes and edges in the graph
print("Total number of graph nodes:", G.number_of_nodes())
print("Total number of graph edges:", G.number_of_edges())
