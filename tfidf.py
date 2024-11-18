import pandas as pd
import math
from collections import Counter

# Load the preprocessed training data
bias_train = pd.read_csv('preprocessed_bias_train.csv')

# Ensure 'lemmatized_tokens' is loaded as a list (convert from string representation if needed)
bias_train['lemmatized_tokens'] = bias_train['lemmatized_tokens'].apply(eval)

# Custom TF, IDF, and TF-IDF computation functions
def compute_term_frequency(row_tokens):
    """
    Compute Term Frequency (TF) for a row of tokens.
    """
    term_frequency = Counter(row_tokens)  # Count occurrences of each word
    total_token_count = len(row_tokens)  # Total number of words in the row
    # Compute relative frequency of each word
    term_frequency = {word: count / total_token_count for word, count in term_frequency.items()}
    return term_frequency

def compute_inverse_document_frequency(corpus):
    """
    Compute Inverse Document Frequency (IDF) for the entire corpus.
    """
    number_of_rows = len(corpus)
    row_frequency = Counter()  # Record how many rows each word appears in
    # Loop over each row in the corpus
    for row_tokens in corpus:
        unique_tokens_in_row = set(row_tokens)  # Get unique words in the row
        for word in unique_tokens_in_row:  # Count words appearing at least once in the row
            row_frequency[word] += 1
    # Compute IDF for each word
    inverse_document_frequency = {
        word: math.log(number_of_rows / (1 + row_count))  # Add 1 to prevent division by zero
        for word, row_count in row_frequency.items()
    }
    return inverse_document_frequency

def compute_term_frequency_inverse_document_frequency(corpus):
    """
    Compute Term Frequency-Inverse Document Frequency (TF-IDF) for the entire corpus.
    """
    # Compute IDF for the whole corpus
    inverse_document_frequency = compute_inverse_document_frequency(corpus)
    tfidf_scores_for_rows = []
    # Loop over each row in the corpus
    for row_tokens in corpus:
        # Compute the term frequency for the row
        term_frequency = compute_term_frequency(row_tokens)
        # Compute TF-IDF for each word in the row
        tfidf = {
            word: term_frequency.get(word, 0) * inverse_document_frequency.get(word, 0)
            for word in term_frequency
        }
        tfidf_scores_for_rows.append(tfidf)
    return tfidf_scores_for_rows

# Step 1: Extract the corpus (list of tokenized rows)
corpus = bias_train['lemmatized_tokens']

# Step 2: Compute TF-IDF using the custom implementation
tfidf_scores = compute_term_frequency_inverse_document_frequency(corpus)

# Step 3: Add the TF-IDF scores to the DataFrame
bias_train['tfidf_features'] = tfidf_scores

# Step 4: Save the TF-IDF features to a new CSV file
bias_train[['text', 'tfidf_features']].to_csv('tfidf_bias_train_custom.csv', index=False)

# Preview the output
print(bias_train[['text', 'tfidf_features']].head())
