import pandas as pd
import numpy as np
import random
from collections import Counter

# Load the preprocessed bias data and aspects data
bias_data = pd.read_csv('preprocessed_bias_train.csv')
aspects_data = pd.read_csv('preprocessed_aspects.csv')

# Ensure the 'lemmatized_tokens' column is parsed as a list
bias_data['lemmatized_tokens'] = bias_data['lemmatized_tokens'].apply(eval)
aspects_data['lemmatized_tokens'] = aspects_data['lemmatized_tokens'].apply(eval)

# Prepare the corpus for LDA (bias-related text)
bias_corpus = bias_data['lemmatized_tokens'].tolist()

# Custom LDA class (already provided)
class CustomLDA:
    def __init__(self, corpus, num_topics=50, alpha=0.01, beta=0.1, num_iterations=100):
        self.corpus = corpus
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations
        self.vocab = set(word for doc in corpus for word in doc)
        self.vocab_size = len(self.vocab)
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.doc_topic_counts = np.zeros((len(corpus), num_topics))
        self.topic_word_counts = np.zeros((num_topics, self.vocab_size))
        self.topic_counts = np.zeros(num_topics)
        self.word_assignments = []
        for doc_idx, doc in enumerate(corpus):
            word_assignments_doc = []
            for word in doc:
                word_id = self.word_to_id[word]
                topic = random.randint(0, num_topics - 1)
                word_assignments_doc.append(topic)
                self.doc_topic_counts[doc_idx, topic] += 1
                self.topic_word_counts[topic, word_id] += 1
                self.topic_counts[topic] += 1
            self.word_assignments.append(word_assignments_doc)

    def sample_topic(self, doc_idx, word_idx, word_id):
        current_topic = self.word_assignments[doc_idx][word_idx]
        self.doc_topic_counts[doc_idx, current_topic] -= 1
        self.topic_word_counts[current_topic, word_id] -= 1
        self.topic_counts[current_topic] -= 1
        topic_probs = (self.doc_topic_counts[doc_idx] + self.alpha) * \
                      (self.topic_word_counts[:, word_id] + self.beta) / \
                      (self.topic_counts + self.beta * self.vocab_size)
        topic_probs /= topic_probs.sum()
        new_topic = np.random.choice(self.num_topics, p=topic_probs)
        self.doc_topic_counts[doc_idx, new_topic] += 1
        self.topic_word_counts[new_topic, word_id] += 1
        self.topic_counts[new_topic] += 1
        return new_topic

    def train(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            for doc_idx, doc in enumerate(self.corpus):
                for word_idx, word in enumerate(doc):
                    word_id = self.word_to_id[word]
                    new_topic = self.sample_topic(doc_idx, word_idx, word_id)
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

# Train the LDA model on bias data
lda = CustomLDA(bias_corpus, num_topics=50, num_iterations=100)
lda.train()

# Print the top words for each topic
print("\nIdentified Topics:")
lda.print_topics(top_n=5)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Map LDA topics to known aspects using cosine similarity
def map_topic_to_aspects(top_words, aspects_data):
    # Initialize a CountVectorizer to convert text to a vector of word counts
    vectorizer = CountVectorizer()
    
    # Prepare the list of aspect keywords as text (for similarity calculation)
    aspect_texts = [' '.join(aspect['lemmatized_tokens']) for _, aspect in aspects_data.iterrows()]
    
    # Convert top words of the topic into a single string
    topic_text = ' '.join(top_words)
    
    # Combine the aspect texts and the topic text to calculate cosine similarity
    texts = aspect_texts + [topic_text]
    
    # Fit and transform the text data into a document-term matrix
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Compute the cosine similarity between the topic and each aspect
    similarity_matrix = cosine_similarity(doc_term_matrix[-1], doc_term_matrix[:-1])
    
    # Find the aspect with the highest similarity score
    most_similar_aspect_idx = similarity_matrix.argmax()
    
    # Retrieve the aspect with the highest similarity score
    most_similar_aspect = aspects_data.iloc[most_similar_aspect_idx]['Aspect']
    
    return most_similar_aspect

# Map LDA topics to known aspects (using cosine similarity)
topics_mapping = []
for topic_id in range(lda.num_topics):
    top_words = [
        lda.id_to_word[word_id] for word_id in np.argsort(lda.topic_word_counts[topic_id] / lda.topic_counts[topic_id])[::-1][:5]
    ]
    
    # Find the most relevant aspect using cosine similarity
    topic_label = map_topic_to_aspects(top_words, aspects_data)
    
    topics_mapping.append({"topic": topic_label, "keywords": top_words})

# Save the topics mapping to a CSV file
topics_df = pd.DataFrame(topics_mapping)
topics_df.to_csv('lda_topics.csv', index=False)

# Print the mapping
print("\nTopics Mapped to Keywords:")
print(topics_df)