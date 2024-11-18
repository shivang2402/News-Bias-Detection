# Import necessary libraries
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure required NLTK packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the training, validation, and aspects data
bias_train = pd.read_csv('../News-Bias-Detection/BEAD/1-Text-Classification/bias-train.csv')
bias_valid = pd.read_csv('../News-Bias-Detection/BEAD/1-Text-Classification/bias-valid.csv')
aspects = pd.read_csv('../News-Bias-Detection/BEAD/3-Aspects/aspects.csv')


# Step 1: Define preprocessing functions
def clean_text(text):
    """
    Clean the input text by:
    - Removing special characters, numbers, and punctuation.
    - Converting text to lowercase.
    - Removing extra whitespaces.
    """
    if pd.isnull(text):  # Handle missing values
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lemmatize a list of tokens to reduce words to their root forms.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

# Step 2: Apply preprocessing steps to the datasets
for df in [bias_train, bias_valid, aspects]:
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Tokenize text
    df['tokens'] = df['cleaned_text'].apply(word_tokenize)
    
    # Remove stopwords
    df['tokens_without_stopwords'] = df['tokens'].apply(remove_stopwords)
    
    # Lemmatize tokens
    df['lemmatized_tokens'] = df['tokens_without_stopwords'].apply(lemmatize_tokens)

# Check for missing values and drop if necessary
print("Missing values in bias_train before cleanup:", bias_train.isnull().sum())
bias_train = bias_train.dropna()
print("Missing values in bias_valid before cleanup:", bias_valid.isnull().sum())
bias_valid = bias_valid.dropna()
print("Missing values in aspects before cleanup:", aspects.isnull().sum())
aspects = aspects.dropna()

# Save the preprocessed datasets
bias_train.to_csv('preprocessed_bias_train.csv', index=False)
bias_valid.to_csv('preprocessed_bias_valid.csv', index=False)
aspects.to_csv('preprocessed_aspects.csv', index=False)

# Preview preprocessed data
print("Preprocessed Training Data:\n", bias_train.head())
print("Preprocessed Validation Data:\n", bias_valid.head())
print("Preprocessed Aspects Data:\n", aspects.head())
