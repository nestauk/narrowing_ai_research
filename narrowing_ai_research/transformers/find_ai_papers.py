import logging
import pickle
import random
import pandas as pd
import numpy as np
import json
import os
import yaml

import narrowing_ai_research
from narrowing_ai_research.utils.list_utils import flatten_freq

project_dir = narrowing_ai_research.project_dir

def get_salient_terms(
    text, category, cat_set, corpus_cat, cat_df, occurrences=1000, number=20
):
    """Extract salient terms from categories in a corpus
    Args:
        text: corpus of text
        category: category for which we want to identify salient terms
        cat_set: grouped df where the keys are categories and the values
            are lists of papers in the category
        corpus_cat: is the category to which the corpus belongs
        cat_df: corpus with all categories

    Returns:
        A df with salient terms in the category
    """
    # Get salient words in corpus (for normalisation)
    logging.info("making corpus salient")
    cats_ = cat_df.copy()

    # This creates a list of ids for papers in the corpus
    corpus = (
        cats_.assign(cat=lambda x: [corpus_cat in a for a in x["category_id"]])
        .groupby("cat")["article_id"]
        .apply(lambda x: set(x))[True]
    )

    # Top terms in the corpus (all papers in category)
    corpus_salient = flatten_freq(text.loc[corpus]["tokenised"])

    # Top terms in the category
    logging.info(f"making {category} salient")
    category_salient = flatten_freq(text.loc[(cat_set[category])]["tokenised"])

    # Extract terms
    logging.info("normalising")
    category_norm = (
        pd.concat([category_salient, corpus_salient], axis=1)
        .rename(columns={0: "category", 1: "corpus"})
        .assign(norm=lambda x: x["category"] / x["corpus"])
        .query(f"category>{occurrences}")
        .sort_values("norm", ascending=False)[:number]
    )

    return category_norm

def count_terms(arx_df, paper_ids, terms):
    """Count terms occurrences in the AI corpus
    Args:
        arx_df: arxiv df
        paper_ids: AI paper IDs
        terms: terms for which we want frequencies
    """
    subset = arx_df.loc[paper_ids]

    terms_n = {
        _id: sum(x in row["tokenised"] for x in terms) for _id, row in subset.iterrows()
    }

    return terms_n

def get_expanded_vocabulary(
    text,
    cat_sets,
    cats,
    category,
    corpus_category,
    w2v,
    expansion_thres=0.5,
    expansion_n=30,
):
    """Expand a vocabulary of salient terms
    Args:
        text: corpus
        cat_sets: grouped object with papers in category
        cats: ids to categories lookup
        category: the arXiv category for which we want to expand terms
        corpus_category: the broader category where the category sits
        w2v: model used for keyword expansion
        expansion_thres: minimum distance to salient term for inclusion
        expansion_n: size of the expansion
    """
    # Get salient terms
    sal = get_salient_terms(
        text=text,
        category=category,
        cat_set=cat_sets,
        corpus_cat=corpus_category,
        cat_df=cats,
    )
    # Expand them
    expanded = set(
        [
            x[0]  # This is the term
            for x in w2v.wv.most_similar(sal.index, topn=expansion_n)
            if x[1] > expansion_thres
        ]
        + list(sal.index)  # Plus the salient terms
    )
    return expanded

def get_expanded_papers(
    arx,
    category,
    cat_sets,
    expanded_terms,
    expansion_value=1,
    random_sample=[False, None],
):
    """Expands papers from the initial category
    Args:
        arx: arxiv papers
        category: category to expand
        cat_sets: grouped object with paper ids by category
        expanded_terms: list of terms in the expanded list
        expansion_value: critical value to set threshold for expansion
        random_sample: whether we calculate term presence in a random set of
            papers outside the category or the full corpus
    Returns:
        list of expanded IDs + a list of relevant terms in the category papers
            and outside the category papers (used to calculate means)
    """
    logging.info("Counting terms in category")
    in_terms = count_terms(arx, list(cat_sets[category]), expanded_terms)
    in_term_values = list(in_terms.values())

    # Mean of expanded term occurrence in the category
    logging.info(np.mean(in_term_values))

    logging.info("Counting terms outside category")

    # Are we working with a random sample or everything?
    if True in random_sample:
        out_paper_ids = random.sample(
            set(arx.index) - cat_sets[category], random_sample[1]
        )
    else:
        out_paper_ids = set(arx.index) - cat_sets[category]

    out_terms = count_terms(arx, out_paper_ids, expanded_terms)

    out_terms_values = list(out_terms.values())

    # Mean of expanded term occurrence outside the category
    logging.info(np.mean(out_terms_values))

    # Critical value we use to select papers outside
    # This will be a value between the in-category mean and out-category mean
    crit_v = np.mean(out_terms_values) + expansion_value * np.std(out_terms_values)

    crit_v_2 = np.ceil(np.mean([crit_v, np.mean(in_term_values)]))

    logging.info(crit_v_2)

    # Return expanded IDs
    sel_ids = [k for k, v in out_terms.items() if v > crit_v_2]

    logging.info(len(sel_ids))

    # Returns both the ids and the term counts
    return (sel_ids, [in_terms, out_terms])

def find_ai_papers():

    if os.path.exists(f"{project_dir}/data/interim/find_ai_outputs.p") is True:
        logging.info("Already found AI papers")
    else:
        # This dict contains values to expand search in different categories
        with open(f"{project_dir}/model_config.yaml", "r") as infile:
            expansion_dict = yaml.safe_load(infile)["expansion_dict"]

        logging.info("Read data")

        cats = pd.read_csv(
            f"{project_dir}/data/raw/arxiv_article_categories.csv",
            dtype={"article_id": str},
        )
        text = pd.read_csv(
            f"{project_dir}/data/raw/arxiv_articles.csv",
            dtype={"article_id": str},
            usecols=["article_id", "abstract"],
        )

        with open(f"{project_dir}/data/interim/arxiv_tokenised.json", "r") as infile:
            arxiv_tokenised = json.load(infile)

        text["tokenised"] = text["article_id"].map(arxiv_tokenised)

        with open(f"{project_dir}/models/word2vec.p", "rb") as infile:
            w2v = pickle.load(infile)

        logging.info("Processing data")
        # Create category sets
        ai_cats = ["cs.AI", "cs.NE", "stat.ML", "cs.LG"]
        cat_sets = cats.groupby("category_id")["article_id"].apply(lambda x: set(x))

        # Create one hot encodings
        ai_binary = pd.DataFrame(index=set(cats["article_id"]), columns=ai_cats)

        for c in ai_binary.columns:
            print(c)
            ai_binary[c] = [x in cat_sets[c] for x in ai_binary.index]

        text.set_index("article_id", inplace=True)

        # We remove papers without abstracts and arXiv categories
        # Note: we are using cs.AI as an example - if it is missing then all other
        # categories will be missing too
        arx = pd.concat([ai_binary, text], axis=1).dropna(
            axis=0, subset=["abstract", "cs.AI"]
        )

        logging.info("Finding papers")

        paper_results = {}
        term_counts = {}

        ev_terms_dict = {"cs.AI": [], "cs.NE": [], "cs.LG": [], "stat.ML": []}

        for cat, corp in zip(ai_cats, ["cs.", "cs.", "stat.", "cs."]):

            logging.info(cat)

            logging.info("Expanding vocabulary")
            ev = get_expanded_vocabulary(text, cat_sets, cats, cat, corp, w2v)
            logging.info(ev)

            ev_terms_dict[cat] = ev

            logging.info("Extracting papers")
            ep = get_expanded_papers(
                arx,
                cat,
                cat_sets,
                ev,
                expansion_value=expansion_dict[cat],
                random_sample=[False, None],
            )
            paper_results[cat] = ep[0]
            term_counts[cat] = ep[1]

            print("\n")

        logging.info("Saving results")
        with open(f"{project_dir}/data/interim/find_ai_outputs.p", "wb") as outfile:
            pickle.dump([paper_results, term_counts], outfile)

        with open(f"{project_dir}/data/interim/ai_vocabularies.json", "w") as outfile:
            json.dump(ev_terms_dict, outfile)

if __name__ == "__main__":
    find_ai_papers()
