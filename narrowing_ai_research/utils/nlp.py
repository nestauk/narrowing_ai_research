import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import models
from narrowing_ai_research.utils.list_utils import flatten_freq

stop_words = set(
    stopwords.words("english") + list(string.punctuation) + ["\\n"] + ["quot"]
)

regex_str = [
    r"http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|" r"[!*\(\),](?:%[0-9a-f][0-9a-f]))+",
    r"(?:\w+-\w+){2}",
    r"(?:\w+-\w+)",
    r"(?:\\\+n+)",
    r"(?:@[\w_]+)",
    r"<[^>]+>",
    r"(?:\w+'\w)",
    r"(?:[\w_]+)",
    r"(?:\S)",
]

# Create the tokenizer which will be case insensitive and will ignore space.
tokens_re = re.compile(r"(" + "|".join(regex_str) + ")", re.VERBOSE | re.IGNORECASE)

def tokenize_document(text, remove_stops=False):
    """Preprocess a whole raw document.
    Args:
        text (str): Raw string of text.
        remove_stops (bool): Flag to remove english stopwords
    Return:
        List of preprocessed and tokenized documents
    """
    return [
        clean_and_tokenize(sentence, remove_stops)
        for sentence in nltk.sent_tokenize(text)
    ]

def clean_and_tokenize(text, remove_stops):
    """Preprocess a raw string/sentence of text.
    Args:
       text (str): Raw string of text.
       remove_stops (bool): Flag to remove english stopwords

    Return:
       tokens (list, str): Preprocessed tokens.
    """
    tokens = tokens_re.findall(text)
    _tokens = [t.lower() for t in tokens]
    filtered_tokens = [
        token.replace("-", "_")
        for token in _tokens
        if not (remove_stops and len(token) <= 2)
        and (not remove_stops or token not in stop_words)
        and not any(x in token for x in string.digits)
        and any(x in token for x in string.ascii_lowercase)
    ]
    return filtered_tokens


def make_ngram(tokenised_corpus, n_gram=2, threshold=10):
    """Extract bigrams from tokenised corpus
    Args:
        tokenised_corpus (list): List of tokenised corpus
        n_gram (int): maximum length of n-grams. Defaults to 2 (bigrams)
        threshold (int): min number of n-gram occurrences before inclusion
    Returns:
        ngrammed_corpus (list)
    """

    tokenised = tokenised_corpus.copy()
    t = 1
    # Loops while the ngram length less / equal than our target
    while t < n_gram:
        phrases = models.Phrases(tokenised, threshold=threshold)
        bigram = models.phrases.Phraser(phrases)
        tokenised = bigram[tokenised]
        t += 1
    return list(tokenised)

def salient_words_per_category(token_df, corpus_freqs, thres, top_words=100):
    """Create a list of salient terms in a sub-corpus (normalised by corpus
    frequency).
    Args:
        tokens (list or series): List where every element is a tokenised doc
        corpus_freqs (df): frequencies of terms in the whole corpus
        thres (int): number of occurrences of a term in the subcorpus
        top_words (int): number of salient words to output

    #Returns:
        A df where every element is a term with its salience
    """
    # Create subcorpus frequencies
    subcorpus_freqs = flatten_freq(token_df)
    # Merge with corpus freqs
    merged = pd.concat([pd.DataFrame(subcorpus_freqs), corpus_freqs], 
                       axis=1, sort=True)
    # Normalise
    merged["salience"] = merged.iloc[:, 0] / merged.iloc[:, 1]
    # Filter
    results = (
        merged.loc[merged.iloc[:, 0] > thres]
        .sort_values("salience", ascending=False)
        .iloc[:top_words]
    )
    results.columns = ["sub_corpus", "corpus", "salience"]
    return results