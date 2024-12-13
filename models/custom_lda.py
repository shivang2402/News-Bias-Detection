import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the preprocessed bias data and aspects data (already saved as CSVs)
bias_data = pd.read_csv('tfidf_bias_train_custom.csv')  # CSV with TF-IDF values for the bias data
aspects_data = pd.read_csv('tfidf_aspects_custom.csv')  # CSV with TF-IDF values for the aspects data

# Convert the TF-IDF DataFrame into the required format for the LDA
def convert_tfidf_to_corpus(df):
    corpus = []
    for index, row in df.iterrows():
        # Extract the TF-IDF dictionary
        doc_tfidf = eval(row['tfidf_features'])  # Use eval to convert the string representation of the dict back into a dictionary
        
        # Flatten the dictionary into a form suitable for LDA (no nested structure)
        corpus.append(doc_tfidf)
    return corpus



# Convert the TF-IDF DataFrames into the required format for the LDA
bias_corpus = convert_tfidf_to_corpus(bias_data)
aspects_corpus = convert_tfidf_to_corpus(aspects_data)

# Custom LDA class (modified to handle TF-IDF data)
class CustomLDA:
    def __init__(self, corpus, num_topics=5, alpha=0.01, beta=0.01, num_iterations=10):
        self.corpus = corpus
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations
        self.vocab = set(word for doc in corpus for word in doc.keys())
        self.vocab_size = len(self.vocab)
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.doc_topic_counts = np.zeros((len(corpus), num_topics))
        self.topic_word_counts = np.zeros((num_topics, self.vocab_size))
        self.topic_counts = np.zeros(num_topics)
        self.word_assignments = []

        # Initialize topic assignments for words in each document
        for doc_idx, doc in enumerate(corpus):
            word_assignments_doc = []
            for word, tfidf_score in doc.items():
                word_id = self.word_to_id[word]
                topic = np.random.randint(0, num_topics)
                word_assignments_doc.append(topic)
                self.doc_topic_counts[doc_idx, topic] += tfidf_score
                self.topic_word_counts[topic, word_id] += tfidf_score
                self.topic_counts[topic] += tfidf_score
            self.word_assignments.append(word_assignments_doc)

    def sample_topic(self, doc_idx, word_idx, word_id, tfidf_score):
        current_topic = self.word_assignments[doc_idx][word_idx]
        self.doc_topic_counts[doc_idx, current_topic] -= tfidf_score
        self.topic_word_counts[current_topic, word_id] -= tfidf_score
        self.topic_counts[current_topic] -= tfidf_score
        topic_probs = (self.doc_topic_counts[doc_idx] + self.alpha) * \
                      (self.topic_word_counts[:, word_id] + self.beta) / \
                      (self.topic_counts + self.beta * self.vocab_size)
        topic_probs /= topic_probs.sum()
        new_topic = np.random.choice(self.num_topics, p=topic_probs)
        self.doc_topic_counts[doc_idx, new_topic] += tfidf_score
        self.topic_word_counts[new_topic, word_id] += tfidf_score
        self.topic_counts[new_topic] += tfidf_score
        return new_topic

    def train(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            for doc_idx, doc in enumerate(self.corpus):
                for word_idx, (word, tfidf_score) in enumerate(doc.items()):
                    word_id = self.word_to_id[word]
                    new_topic = self.sample_topic(doc_idx, word_idx, word_id, tfidf_score)
                    self.word_assignments[doc_idx][word_idx] = new_topic

    def print_topics(self, top_n=5):
        for topic_id in range(self.num_topics):
            word_probs = self.topic_word_counts[topic_id] / self.topic_counts[topic_id]
            top_word_ids = np.argsort(word_probs)[::-1][:top_n]
            top_words = [self.id_to_word[word_id] for word_id in top_word_ids]
            print(f"Topic {topic_id}: {', '.join(top_words)}")

    def get_document_topics(self, top_n=5):
        doc_topic_probs = (self.doc_topic_counts + self.alpha) / \
                          (self.doc_topic_counts.sum(axis=1)[:, None] + self.alpha * self.num_topics)
        top_topics_per_doc = []
        for doc_idx in range(len(self.corpus)):
            topic_probs = doc_topic_probs[doc_idx]
            top_topic_ids = np.argsort(topic_probs)[::-1][:top_n]
            top_topic_probs = topic_probs[top_topic_ids]
            top_topics_per_doc.append([(topic_id, prob) for topic_id, prob in zip(top_topic_ids, top_topic_probs)])
        return top_topics_per_doc

# Train the LDA model on the bias data
lda = CustomLDA(bias_corpus, num_topics=5, num_iterations=10)
lda.train()

# Print the top words for each topic
print("\nIdentified Topics:")
lda.print_topics(top_n=15)

def map_topic_to_aspects(tfidf_topic, aspects_data, lda_vocab):
    # Convert 'tfidf_features' column from string representation to actual dictionary
    aspect_tfidf_matrix = []
    
    # Create a dictionary for fast word lookup from the LDA model vocabulary
    word_to_index = {word: i for i, word in enumerate(lda_vocab)}

    for _, row in aspects_data.iterrows():
        doc_tfidf = eval(row['tfidf_features'])  # Use eval to convert string to dictionary
        tfidf_vector = np.zeros(len(lda_vocab))  # Initialize a zero vector of length equal to the LDA vocabulary
        
        # Populate the TF-IDF vector based on the words in the document
        for word, tfidf_value in doc_tfidf.items():
            if word in word_to_index:  # Only include words that are in the LDA vocabulary
                tfidf_vector[word_to_index[word]] = tfidf_value
        
        aspect_tfidf_matrix.append(tfidf_vector)  # Append the vector to the list
    
    # Convert list to numpy array for cosine similarity calculation
    aspect_tfidf_matrix = np.array(aspect_tfidf_matrix)

    # Calculate cosine similarity between the topic TF-IDF vector and each aspect TF-IDF vector
    similarity_scores = cosine_similarity(tfidf_topic.reshape(1, -1), aspect_tfidf_matrix)
    
    # Find the aspect with the highest similarity score
    most_similar_aspect_idx = similarity_scores.argmax()
    most_similar_aspect = aspects_data.iloc[most_similar_aspect_idx]['Aspect']
    
    return most_similar_aspect


# Limit the vocabulary size (e.g., use the top 500 most frequent words)
top_n_words = 500  # Adjust this value as needed
# Sort vocabulary based on cumulative topic-word probabilities
word_importance = lda.topic_word_counts.sum(axis=0)
sorted_vocab_indices = np.argsort(word_importance)[::-1][:top_n_words]
limited_vocab = [lda.id_to_word[i] for i in sorted_vocab_indices]

# Map LDA topics to known aspects (using cosine similarity on TF-IDF vectors)
topics_mapping = []
for topic_id in range(lda.num_topics):
    # Get the top words for each topic
    top_words = [
        lda.id_to_word[word_id] for word_id in np.argsort(lda.topic_word_counts[topic_id] / lda.topic_counts[topic_id])[::-1][:15]
    ]
    
    # Build a TF-IDF vector for the topic (summing the word vectors based on topic-word distribution)
    tfidf_topic_vector = np.zeros(top_n_words)  # Adjust the size based on the reduced vocabulary
    for word in top_words:
        if word in limited_vocab:
            word_id = limited_vocab.index(word)  # Find the index of the word in the reduced vocabulary
            tfidf_topic_vector[word_id] += lda.topic_word_counts[topic_id, word_id]
    
    # # Normalize the topic vector (important for cosine similarity)
    # tfidf_topic_vector /= np.linalg.norm(tfidf_topic_vector)
    
    # Normalize the topic vector (important for cosine similarity)
    norm = np.linalg.norm(tfidf_topic_vector)
    if norm > 0:
        tfidf_topic_vector /= norm
    else:
    # Handle the case where the vector norm is zero (no valid words for this topic)
        tfidf_topic_vector = np.zeros_like(tfidf_topic_vector)  # Set it to a zero vector or handle differently

    # Map the topic to the most relevant aspect
    topic_label = map_topic_to_aspects(tfidf_topic_vector, aspects_data, limited_vocab)
    
    topics_mapping.append({"topic": topic_label, "keywords": top_words})

# Save the topics mapping to a CSV file
topics_df = pd.DataFrame(topics_mapping)
topics_df.to_csv('CUSTOM LDA TOPICS/lda_topics.csv', index=False)

# Print the mapping
print("\nTopics Mapped to Keywords:")
print(topics_df)
