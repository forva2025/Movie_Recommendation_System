# import necessary libraries and packages
import pandas as pd #used for data analysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#sample movie dataset
data = {'movie_id': [1,2,3,4,5],
        'title': ['The Matrix', 'John Wick', 'The Gogfather', 'Pulp Fiction', 'The Dark Knight'],
        'genre': ['Action, Sci-Fi', 'Action, Thriller', 'Crime, Drama', 'Crime, Drama', 'Action, Crime, Drama']}

#convert the dataset into a dataframe
df= pd.DataFrame(data)

#display the dataset
print(f"Movie Data:{df}")


#define a TF-IDF vectorizer to tranform genre text into vectors
tfidf= TfidfVectorizer(stop_words='english')

#Fit and transform the genre column into a matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(df['genre'])

#compute the cosine similarity index

cosine_sim =cosine_similarity(tfidf_matrix, tfidf_matrix)

#function to recommend movies based on cosine similarity

def get_recommendations(title, cosine_sim=cosine_sim):

    #get the index of the moview that matches the title
    idx = df[df['title'] == title].index[0]

    #get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    #sort the movies based on the similarity scores
    sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #Get the indices of the 3 most similar movies
    sim_scores = sim_scores[1:4] #exclude the movie it'self, so start from 1

    #Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    #Return the title of the most similar moview
    return df['title'].iloc[movie_indices]

#Test the recommendation system with an example

movie_title = 'The Matrix'
recommended_movies = get_recommendations(movie_title)

print(f"\nMovies recommended for '{movie_title}': ")

for movie in recommended_movies:
    print(movie) #print all the recommended movies
