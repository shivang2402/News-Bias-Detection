import pandas as pd
import math
from collections import Counter

# Load the preprocessed bias training data, aspects data, and validation data
bias_train = pd.read_csv('results/Preprocessed Data/preprocessed_bias_train.csv')  # Training data
aspects_data = pd.read_csv('results/Preprocessed Data/preprocessed_bias_train.csv')  # Aspects data
bias_valid = pd.read_csv('results/Preprocessed Data/preprocessed_bias_train.csv')  # Validation data

# Ensure 'lemmatized_tokens' is loaded as a list (convert from string representation if needed)
bias_train['lemmatized_tokens'] = bias_train['lemmatized_tokens'].apply(eval)
aspects_data['lemmatized_tokens'] = aspects_data['lemmatized_tokens'].apply(eval)
bias_valid['lemmatized_tokens'] = bias_valid['lemmatized_tokens'].apply(eval)

# Token filtering function: Remove non-alphabetic and short words
def filter_tokens(tokens):
    """
    Filter tokens to remove non-alphabetic and short words.
    """
    return [token for token in tokens if len(token) > 2 and token.isalpha()]

# Apply the filtering function to the tokenized data
bias_train['lemmatized_tokens'] = bias_train['lemmatized_tokens'].apply(lambda x: filter_tokens(x))
aspects_data['lemmatized_tokens'] = aspects_data['lemmatized_tokens'].apply(lambda x: filter_tokens(x))
bias_valid['lemmatized_tokens'] = bias_valid['lemmatized_tokens'].apply(lambda x: filter_tokens(x))

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

# Define frequency thresholds
min_df = 5  # Minimum documents a token must appear in
max_df = 0.7  # Maximum percentage of documents a token can appear in

def compute_inverse_document_frequency(corpus):
    """
    Compute Inverse Document Frequency (IDF) with frequency thresholds.
    """
    number_of_rows = len(corpus)
    row_frequency = Counter()
    for row_tokens in corpus:
        unique_tokens_in_row = set(row_tokens)
        for word in unique_tokens_in_row:
            row_frequency[word] += 1

    # Apply frequency thresholds
    inverse_document_frequency = {
        word: math.log((number_of_rows + 1) / (1 + row_count))
        for word, row_count in row_frequency.items()
        if row_count >= min_df and (row_count / number_of_rows) <= max_df
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
bias_corpus = bias_train['lemmatized_tokens']
aspect_corpus = aspects_data['lemmatized_tokens']
bias_valid_corpus = bias_valid['lemmatized_tokens']  # Add the validation corpus

# Step 2: Compute TF-IDF using the custom implementation
tfidf_bias_scores = compute_term_frequency_inverse_document_frequency(bias_corpus)
tfidf_aspect_scores = compute_term_frequency_inverse_document_frequency(aspect_corpus)
tfidf_bias_valid_scores = compute_term_frequency_inverse_document_frequency(bias_valid_corpus)  # Compute for validation set

# Step 3: Add the TF-IDF scores to the DataFrames
bias_train['tfidf_features'] = tfidf_bias_scores
aspects_data['tfidf_features'] = tfidf_aspect_scores
bias_valid['tfidf_features'] = tfidf_bias_valid_scores  # Add TF-IDF features to validation data

# Step 4: Save the TF-IDF features to new CSV files, including the label column
bias_train[['text', 'label', 'tfidf_features']].to_csv('TFIDF/tfidf_bias_train_custom.csv', index=False)
aspects_data[['text', 'Aspect', 'tfidf_features']].to_csv('TFIDF/tfidf_aspects_custom.csv', index=False)
bias_valid[['text','label', 'tfidf_features']].to_csv('TFIDF/tfidf_bias_valid_custom.csv', index=False)  # Save validation data

# Preview the output
print("Bias Train Data with TF-IDF and Labels:")
print(bias_train[['text', 'label', 'tfidf_features']].head())
print("\nAspects Data with TF-IDF:")
print(aspects_data[['text', 'Aspect', 'tfidf_features']].head())
print("\nBias Validation Data with TF-IDF:")
print(bias_valid[['text','label', 'tfidf_features']].head())
