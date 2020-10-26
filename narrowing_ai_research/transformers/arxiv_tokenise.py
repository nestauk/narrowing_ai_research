import os
import pandas as pd
import json
import logging
import narrowing_ai_research
from narrowing_ai_research.utils.nlp import clean_and_tokenize, make_ngram

project_dir = narrowing_ai_research.project_dir


def arxiv_tokenise():

    if os.path.exists(f"{project_dir}/data/interim/arxiv_tokenised.json") is True:

        logging.info("Already tokenised data")

    else:
        logging.info("Reading data")
        arxiv_articles = pd.read_csv(
            f"{project_dir}/data/raw/arxiv_articles.csv",
            dtype={"article_id": str},
            usecols=["article_id", "abstract"],
        )

        # Remove papers without abstracts
        arxiv_w_abst = arxiv_articles.dropna(axis=0, subset=["abstract"])

        logging.info("Cleaning and tokenising")
        arxiv_tokenised = [
            clean_and_tokenize(x, remove_stops=True) for x in arxiv_w_abst["abstract"]
        ]

        logging.info("Making ngrams")
        arxiv_ngrams = make_ngram(arxiv_tokenised, n_gram=3)

        # Turn into dictionary mapping ids to token lists
        out = {i: t for i, t in zip(arxiv_w_abst["article_id"], arxiv_ngrams)}

        logging.info("Saving")
        with open(f"{project_dir}/data/interim/arxiv_tokenised.json", "w") as outfile:
            json.dump(out, outfile)


if __name__ == "__main__":
    arxiv_tokenise()
