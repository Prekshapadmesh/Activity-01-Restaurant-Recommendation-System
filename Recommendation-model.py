#Importing Necessary Libraries


import pandas as pd
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading both the datasets
df_percent=pd.read_csv("C:/Users/preks/OneDrive/Desktop/zomato-dataset/processed_data.csv")
zomato=pd.read_csv("C:/Users/preks/OneDrive/Desktop/zomato-dataset/clean_data.csv")
print(df_percent.head())

df_percent = zomato.sample(frac=0.5)
df_percent.shape

df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

##TF-IDF is the statistical method of evaluating the significance of a word in a given document.

#TF — Term frequency(tf) refers to how many times a given term appears in a document.
#IDF — Inverse document frequency(idf) measures the weight of the word in the document, i.e if the word is common or rare in the entire document. 
#The TF-IDF intuition follows that the terms that appear frequently in a document are less important than terms that rarely appear.

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(name, cosine_similarities=cosine_similarities):
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the restaurant entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them by the biggest number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Use pd.concat instead of append to avoid the error
    for each in recommend_restaurant:
        df_new = pd.concat([df_new, df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each].sample()])
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print(f'TOP {len(df_new)} RESTAURANTS LIKE {name} WITH SIMILAR REVIEWS:')
    
    return df_new

# Testing the function with 'Pai Vihar'
recommend('Pai Vihar')




