# Import necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import words
import nltk

CUSTOMSTOPWORDS = ["pisces", "tho", "kinda", "having", "word", "drill", "vista", "having", "command", "weekend", "gettin", "naval", "sarahmascara", "rustyrockets", "script", "eat", "does", "cast", "smt", "have", "highway", "log", "twitter", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "abst", "ac", "across", "act", "ad", "added", "adj", "ae", "af", "after", "afterwards", "ag", "ah", "ain", "ain't", "aj", "al", "all", "along", "also", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "ao", "ap", "appear", "ar", "are", "aren", "arent", "aren't", "arise", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "aw", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "being", "beside", "besides", "between", "bi", "biol", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cause", "causes", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "consider", "considering", "contain", "containing", "contains", "corresponding", "course", "cp", "cq", "cr", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "day", "describe", "described", "detail", "df", "di", "different", "dj", "dk", "dl", "doing", "don", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "ej", "el", "eleven", "else", "em", "en", "eo", "ep", "eq", "er", "es", "est", "et", "et-al", "etc", "eu", "ev", "ex", "example", "ey", "f", "f2", "fa", "far", "fc", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "has", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hey", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "how", "howbeit", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "in", "inasmuch", "inc", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "last", "lb", "lc", "le", "les", "less", "let", "lets", "let's", "lf", "line", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "make", "makes", "many", "may", "me", "means", "meantime", "meanwhile", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "need", "needs", "nevertheless", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "probably", "promptly", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regards", "related", "relatively", "research", "research-articl", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "six", "sixty", "sj", "sl", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "sub", "such", "sup", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tries", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "until", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "with", "within", "wo", "won", "wont", "won't", "words", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz", "wan"]

# Load the datasets
bias_train = pd.read_csv('../News-Bias-Detection/BEAD/1-Text-Classification/bias-train.csv')
bias_valid = pd.read_csv('../News-Bias-Detection/BEAD/1-Text-Classification/bias-valid.csv')
aspects = pd.read_csv('../News-Bias-Detection/BEAD/3-Aspects/aspects.csv')

# Initialize the lemmatizer and valid English words set
lemmatizer = WordNetLemmatizer()
valid_words = set(words.words())

# Helper function to map Treebank POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Helper function to check if a word is valid
def is_valid_word(word):
    return word in valid_words or wordnet.synsets(word)

# Define preprocessing functions
def clean_text(text):
    """
    Clean the input text by:
    - Converting text to lowercase.
    - Removing non-alphabetic characters.
    - Removing extra whitespaces.
    - Removing stopwords using sklearn's stopword list.
    """
    if pd.isnull(text):  # Handle missing values
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove everything except letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    words = text.split()
    filtered_words = [word for word in words if word not in CUSTOMSTOPWORDS]
    return ' '.join(filtered_words)

def lemmatize_tokens(tokens):
    """
    Lemmatize a list of tokens using POS tags and check validity.
    Replace invalid words with "XXXXX".
    """
    pos_tagged = pos_tag(tokens)  # POS tagging
    lemmatized = []
    for token, tag in pos_tagged:
        # Lemmatize with POS tags
        lemma = lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        # Add stricter filtering
        if len(lemma) > 2 and lemma.isalpha() and is_valid_word(lemma):
            lemmatized.append(lemma)
    return lemmatized  # Return only valid tokens


# Apply preprocessing steps to the datasets
for df in [bias_train, bias_valid, aspects]:
    # Clean text
    df['processed_text'] = df['text'].apply(clean_text)
    
    # Tokenize text
    df['tokens'] = df['processed_text'].apply(word_tokenize)
    
    # Lemmatize tokens
    df['lemmatized_tokens'] = df['tokens'].apply(lemmatize_tokens)

# Check for missing values and drop rows with any missing values
print("Missing values in bias_train before cleanup:", bias_train.isnull().sum())
bias_train = bias_train.dropna()
print("Missing values in bias_valid before cleanup:", bias_valid.isnull().sum())
bias_valid = bias_valid.dropna()
print("Missing values in aspects before cleanup:", aspects.isnull().sum())
aspects = aspects.dropna()

# Save the preprocessed datasets
bias_train.to_csv('Preprocessed Data/preprocessed_bias_train.csv', index=False)
bias_valid.to_csv('Preprocessed Data/preprocessed_bias_valid.csv', index=False)
aspects.to_csv('Preprocessed Data/preprocessed_aspects.csv', index=False)

# Preview preprocessed data
print("Preprocessed Training Data:\n", bias_train.head())
print("Preprocessed Validation Data:\n", bias_valid.head())
print("Preprocessed Aspects Data:\n", aspects.head())
