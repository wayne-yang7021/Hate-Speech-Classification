from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import nltk
from collections import defaultdict
import math

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
dictionary = {}

# Load labeled data
# labeled_data = pd.read_csv("data/labeled_data.csv")
labeled_data = pd.read_csv("data/labeled_data.csv")
df = pd.DataFrame(labeled_data)
tweets = df["tweet"]

# Preprocess and tokenize tweets
def preprocess_and_tokenize(tweets):
    term_frequency = []
    token_index = 0
    doc_frequency = defaultdict(int)  # Dictionary to store document frequency for each term
    for tweet in tweets:
        # Remove URLs, mentions, and hashtags
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)  # Remove URLs
        tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
        tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags

        # Remove non-alphanumeric characters and convert to lowercase
        tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet).lower()

        # Split words and remove stopwords
        words = tweet.split()
        filtered_words = [word for word in words if word not in stop_words]

        # Apply stemming
        stemmed_words = [ps.stem(word) for word in filtered_words]

        # Add words to dictionary and count term frequency for used tokens only
        tf_dict = {}
        seen_terms = set()  # Set to track terms seen in this tweet
        for word in stemmed_words:
            if word not in dictionary:
                dictionary[word] = (token_index, 0)  # Store (index, document frequency)
                token_index += 1

            word_index, _ = dictionary[word]  # Retrieve index, df
            if word not in seen_terms:  # Only update document frequency once per tweet
                dictionary[word] = (word_index, dictionary[word][1] + 1)  # Increment DF
                seen_terms.add(word)

            if word_index not in tf_dict:
                tf_dict[word_index] = 0
            tf_dict[word_index] += 1

        # Append term frequency dictionary for the current tweet
        term_frequency.append(tf_dict)

    return term_frequency

# Step 1: Calculate Inverse Document Frequency (IDF)
def calculate_idf(dictionary, total_docs):
    idf = {}
    for term, (index, df) in dictionary.items():
        # Apply log and smoothing (adding 1 to DF to avoid division by zero)
        idf[term] = math.log10(total_docs / (df + 1))  # Adding 1 for smoothing
    return idf


# Step 2: Calculate TF-IDF for each tweet
def calculate_unit_tfidf(term_frequency_vectors, idf, dictionary):
    tfidf_vectors = []
    
    for tf_dict in term_frequency_vectors:
        tfidf_list = []
        
        # For each term in the tweet, calculate its TF-IDF
        for word_index, tf in tf_dict.items():
            # Retrieve the term based on the word index
            term = [k for k, v in dictionary.items() if v[0] == word_index][0]
            
            # Get the corresponding IDF value for the term
            term_idf = idf.get(term, 0)  # Default to 0 if term is not found
            tfidf_value = tf * term_idf  # TF-IDF for the term
            tfidf_list.append((word_index, tfidf_value))  # Store (index, tfidf)
        
        # Sort the TF-IDF list by index in ascending order
        tfidf_list.sort(key=lambda x: x[0])  # Sort by term index
        tfidf_vectors.append(tfidf_list)
    
    # Step 3: Normalize the TF-IDF vectors to unit vectors
    unit_tfidf_vectors = []
    
    for tfidf_list in tfidf_vectors:
        magnitude = math.sqrt(sum(tfidf**2 for _, tfidf in tfidf_list))
        
        if magnitude == 0:  # To avoid division by zero
            unit_tfidf_vectors.append([(index, 0) for index, _ in tfidf_list])
        else:
            unit_tfidf_vector = [(index, tfidf / magnitude) for index, tfidf in tfidf_list]
            unit_tfidf_vectors.append(unit_tfidf_vector)
    
    return unit_tfidf_vectors

def cosine(doc_x_list, doc_y_list):
    i = 0
    j = 0
    sum = 0
    while(i < len(doc_x_list) - 1 and j < len(doc_y_list) - 1):
        if doc_x_list[i][0] == doc_y_list[j][0]:
            sum += doc_x_list[i][1] * doc_y_list[j][1]
            i += 1
            j += 1
        elif doc_x_list[i][0] < doc_y_list[j][0]:
            i += 1
        else:
            j += 1
    return sum


# Tokenize tweets and compute term frequency vectors
term_frequency_vectors = preprocess_and_tokenize(tweets)

total_docs = len(tweets)

idf = calculate_idf(dictionary, total_docs)

unit_tfidf_vectors = calculate_unit_tfidf(term_frequency_vectors, idf, dictionary)

import pickle

# 保存 unit_tfidf_vectors
with open("unit_tfidf_vectors.pkl", "wb") as f:
    pickle.dump(unit_tfidf_vectors, f)

# 保存 dictionary
with open("dictionary.pkl", "wb") as f:
    pickle.dump(dictionary, f)

print("Dictionary with key: token,  value: (index, df):", dictionary)
print("Term Frequency Vectors (dictionary index num, tf):", term_frequency_vectors[89])
print("Unit TF-IDF Vectors (dictionary index num, unit tf-idf value):", unit_tfidf_vectors[89])
print("Cosine similarity of tweet 89 and 90:", cosine(unit_tfidf_vectors[89], unit_tfidf_vectors[90]))

