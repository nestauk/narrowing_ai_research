# Robustness utilities

import tomotopy as tp
import numpy as np
import pandas as pd


def train_tomotopy_model(corpus: dict, num_topics: int = 300, verbose: bool = True):

    mdl = tp.LDAModel(k=num_topics)

    corpus_ = {k: v for k, v in corpus.items() if len(v) > 0}

    for doc in list(corpus_.values()):
        mdl.add_doc(doc)

    for i in range(0, 150, 10):
        mdl.train(10)
        print("Iteration: {}".format(i))

    if verbose is True:
        print(mdl.summary())

    return mdl


def get_topic_words(topic, top_words=5):
    """Extracts main words for a topic"""

    return "_".join([x[0] for x in topic[:top_words]])


def make_topic_mix(mdl, num_topics, doc_indices):
    """Takes a tomotopy model and products a topic mix"""
    topic_mix = pd.DataFrame(
        np.array([mdl.docs[n].get_topic_dist() for n in range(len(doc_indices))])
    )

    topic_mix.columns = [
        get_topic_words(mdl.get_topic_words(n, top_n=5)) for n in range(num_topics)
    ]

    topic_mix.index = doc_indices
    return topic_mix
