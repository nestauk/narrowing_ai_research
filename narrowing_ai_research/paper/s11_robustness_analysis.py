# Script to assess robustness of diversity metrics

import altair as alt
import json
import yaml
import logging
import numpy as np
import pandas as pd
from narrowing_ai_research import project_dir
from narrowing_ai_research.utils.read_utils import (
    get_ai_ids,
    read_tokenised,
    read_papers,
)
from narrowing_ai_research.paper.rob_utils import make_topic_mix, train_tomotopy_model
from itertools import chain
from toolz import pipe
from numpy.random import choice
from functools import partial

from narrowing_ai_research.paper.s4_diversity_macro import year_diversity_results

# Functions


def get_diversity_config():
    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        return yaml.safe_load(infile)["section_4"]["div_params"]


def sample_years(
    papers: pd.DataFrame, n: int = 1000, years: range = range(2010, 2021)
) -> list:
    """Samples n papers from each year to remove biases"""

    article_ids = []
    for y in years:
        article_ids.append(papers.query(f"year=={y}").sample(n)["article_id"].tolist())
    return set(chain(*article_ids))


def select_tokenised(selected_ids: set, tok: dict):
    """Selects tokenised variables leaving out empty abstracts"""

    return {k: v for k, v in tok.items() if k in selected_ids if len(v) > 0}


def get_semantic_results():
    with open(f"{project_dir}/data/raw/ai_semantic_results.json", "r") as infile:
        sr = json.load(infile)

    # Remove a few "none" semantic scholar results
    NoneType = type(None)
    return [art for art in sr if type(art) != NoneType]


def get_citation_meta(article: dict, direction: str = "citations") -> dict:

    return article[direction]


def get_citation_ids(semantic_results, direction: str = "citations", cit_n: int = 1):
    """Returns ids for AI paper citations and
    references below a threshold and above a citation threshold"""

    cit_df = pd.DataFrame(
        chain(*[get_citation_meta(art, direction) for art in semantic_results])
    )[["arxivId", "year"]].query("year<=2020")["arxivId"]

    cit_counts = cit_df.value_counts()

    return pipe(cit_counts.loc[cit_counts > cit_n].index, set)


def get_expanded_ai_ids():
    """Expands AI ids with highly"""

    ai_ids = get_ai_ids()
    semantic_results = get_semantic_results()

    citation_ids, reference_ids = [
        get_citation_ids(semantic_results, direction)
        for direction in ["citations", "references"]
    ]

    sampled_ai_ids = set(
        choice(list(ai_ids), size=int(np.mean([len(citation_ids), len(reference_ids)])))
    )

    return sampled_ai_ids | citation_ids | reference_ids


def estimate_diversity(
    papers,
    it: int = 0,
    sample_size: int = 1000,
    num_topics: int = 300,
    range_topics: range = None,
) -> pd.DataFrame:
    """Estimate diversity on a sample
    Args:
        papers: papers to use in the test
        it: iteration number
        sample_size: number of papers to sample (if None we use all of them)
        num_topics: number of topics to fit in the topic model
        range_topics: range within which to select a topic
    """

    if range_topics is not None:
        num_topics = int(choice(range_topics, 1))

    logging.info(f"iteration {it}")
    logging.info(f"Training and fitting topic model with {num_topics} topics")

    if sample_size is not None:
        selected_corpus = pipe(
            papers,
            partial(sample_years, n=sample_size),
            partial(select_tokenised, tok=tok),
        )
    else:
        selected_corpus = select_tokenised(set(papers["article_id"]), tok)

    topic_mix = pipe(
        selected_corpus,
        partial(train_tomotopy_model, num_topics=num_topics, verbose=False),
        partial(
            make_topic_mix, num_topics=num_topics, doc_indices=selected_corpus.keys()
        ),
    )

    year_ids = (
        papers.loc[papers["article_id"].isin(selected_corpus.keys())]
        .groupby("year")["article_id"]
        .apply(lambda x: set(x))
    )

    logging.info("Calculating normalised diversity metrics")
    return year_diversity_results(topic_mix, year_ids, div_params, t0=2010).assign(
        iteration=it
    )


def get_diversity_df(diversity_results, wind=2):
    """Combines diversity results and calculates rolling window"""

    return (
        pd.concat(diversity_results)
        .groupby(["diversity_metric", "iteration", "parametre_set"])
        .apply(
            lambda df: df.set_index("year")
            .rolling(window=wind)
            .mean()
            .drop(axis=1, labels=["iteration"])
        )
        .reset_index(drop=False)
        .dropna()
    )


def plot_diversity_bootstrapped(diversity_df):
    """Plots the result of bootstrapped diversity"""

    div_lines = (
        alt.Chart()
        .mark_line()
        .encode(
            x="year:O",
            y=alt.Y("mean(score)", scale=alt.Scale(zero=False)),
            color="parametre_set",
        )
    )

    div_bands = (
        alt.Chart()
        .mark_errorband(extent="ci")
        .encode(
            x="year:O",
            y=alt.Y("score", scale=alt.Scale(zero=False)),
            color="parametre_set",
        )
    )

    out = alt.layer(
        div_lines, div_bands, data=diversity_df, height=150, width=400
    ).facet(row="diversity_metric", column="test")

    return out


if __name__ == "__main__":

    NUM_RUNS = 20
    LDA_SAMPLE = 50000

    logging.info(f"Running robustness analysis with {NUM_RUNS} runs")

    papers = read_papers()

    papers_ai = papers.query("is_ai==True").reset_index(drop=True)

    tok = read_tokenised()

    div_params = get_diversity_config()

    logging.info("Calculating time adjusted diversity")

    diversity_time_df = pipe(
        [estimate_diversity(papers_ai, it=n) for n in range(NUM_RUNS)], get_diversity_df
    )
    diversity_time_df.to_csv(
        f"{project_dir}/data/processed/robustness_time.csv", index_label=False
    )

    logging.info("Calculating LDA diversity")
    diversity_lda_df = pipe(
        [
            estimate_diversity(
                papers.sample(n=LDA_SAMPLE),
                it=n,
                sample_size=None,
                range_topics=range(200, 400),
            )
            for n in range(NUM_RUNS)
        ],
        get_diversity_df,
    )
    diversity_lda_df.to_csv(
        f"{project_dir}/data/processed/robustness_lda.csv", index_label=False
    )

    logging.info("Calculating expanded corpus diversity")

    expanded_ids = get_expanded_ai_ids()

    papers_expanded = papers.loc[papers["article_id"].isin(expanded_ids)].reset_index(
        drop=True
    )

    diversity_expanded_df = pipe(
        [
            estimate_diversity(
                papers_expanded,
                it=n,
                sample_size=None,
                range_topics=range(200, 400),
            )
            for n in range(NUM_RUNS)
        ],
        get_diversity_df,
    )

    diversity_expanded_df.to_csv(
        f"{project_dir}/data/processed/robustness_expanded.csv", index_label=False
    )
