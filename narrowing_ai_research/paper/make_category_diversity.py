import pandas as pd
import logging
import yaml
from scipy.spatial.distance import pdist, squareform

from narrowing_ai_research.transformers.diversity import Diversity, remove_zero_axis

from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_papers_orgs,
    read_topic_mix,
    paper_orgs_processing
)

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir


def read_process_data():
    papers = read_papers()
    paper_orgs = paper_orgs_processing(read_papers_orgs(), papers)
    paper_orgs["year"] = [x.year for x in paper_orgs["date"]]

    topic_mix = read_topic_mix()
    topic_mix.set_index("article_id", inplace=True)

    return papers, paper_orgs, topic_mix


def create_ids(df, paper_threshold, year_threshold, variable="org_name"):
    """Creates a list of paper sets for top entities"""

    df_ = df.loc[df["year"] > year_threshold]

    counts = df_.loc[df_["is_ai"] == True][variable].value_counts()

    tops = counts.loc[counts > paper_threshold].index

    sets = (
        df_.loc[df[variable].isin(tops)]
        .groupby(variable)["article_id"]
        .apply(lambda x: set(x))
    )

    return sets


def make_rao_stirling_matrices(topics, div_params):
    """Returns distance matrices for different parametre sets"""
    rao_stirling_matrices = {}

    for n, v in enumerate(div_params["rao_stirling"]):
        logging.info(n)
        tf = remove_zero_axis(topics)
        dist = pd.DataFrame(
            squareform(pdist(tf.T, metric=v["distance"])),
            columns=tf.columns,
            index=tf.columns,
        )

        rao_stirling_matrices[n] = dist
    return rao_stirling_matrices


def calculate_diversity_entity(data, div_params, distance_matrices, name):

    results = []
    # Here we remove axes with zero values
    data_ = remove_zero_axis(data)

    # For each metric, varams key value pair
    for d in div_params.keys():
        for n, b in enumerate(div_params[d]):
            div = Diversity(data_, name)
            if d != "rao_stirling":
                getattr(div, d)(b)
            # Add the distance matrix if we are calculating org diversity
            else:
                getattr(div, d, distance_matrices[n])(b)

            r = [
                name,
                div.metric_name,
                div.metric,
                div.param_values,
                f"Parametre Set {n}",
            ]
            results.append(r)

    return pd.DataFrame(
        results, columns=["category", "metric", "score", "parametres", "parametre_set"]
    )


def make_all_diversities(
    topic_mix, year_sets, category_sets, div_params, rao_stirling_matrices, y0, y1
):
    """Calculate organisational diversities"""
    results = []

    for y in range(y0, y1 + 1):
        logging.info(y)

        for n, o in enumerate(category_sets.keys()):
            logging.info(o)
            o_y = year_sets[y].intersection(category_sets[o])
            print(len(o_y))
            tm_cat = topic_mix.loc[topic_mix.index.isin(o_y)]
            divs = calculate_diversity_entity(
                tm_cat, div_params, rao_stirling_matrices, o
            )
            divs["year"] = y
            results.append(divs)

    return pd.concat(results)


def diversity_estimation(
    paper_orgs, div_params, section_pars, papers, topic_mix, cat="org"
):
    """Calculates diversity based on a category (orgs, countries...)"""

    cat_pars = section_pars[cat]

    # Create a sets of papers for each orgs with threshold = 100
    org_sets = create_ids(
        paper_orgs, cat_pars["paper_threshold"], cat_pars["year_threshold"],
        variable=cat
    )

    year_sets = (
        papers.loc[papers["is_ai"] == True]
        .groupby("year")["article_id"]
        .apply(lambda x: set(x))
    )

    if cat == "org":
        if cat_pars["google_filter"] is True:
            # Create a separate Google set
            org_sets["Google_wo_DeepMind"] = org_sets["Google"] - org_sets["DeepMind"]

    # Identify recent papers
    recent_paper_ids = set(
        papers.loc[papers.year >= cat_pars["year_threshold"]]["article_id"]
    )

    # Identify recent matrix
    topics_recent = topic_mix.loc[topic_mix.index.isin(recent_paper_ids)]

    logging.info("Making matrix")
    rao_stirling_matrices = make_rao_stirling_matrices(topics_recent, div_params)

    logging.info(f"Calculating {cat} diversities")
    div_df = make_all_diversities(
        topic_mix,
        year_sets,
        org_sets,
        div_params,
        rao_stirling_matrices,
        cat_pars["y0"],
        cat_pars["y1"],
    )
    return div_df


# We use the same diversity parametres as in the analysis of diversity
def make_category_diversity(category="org"):

    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        pars = yaml.safe_load(infile)

    div_params = pars["section_4"]["div_params"]
    section_pars = pars["section_8"]

    papers, paper_orgs, topic_mix = read_process_data()

    df = diversity_estimation(
        paper_orgs, div_params, section_pars, papers, topic_mix, cat=category
    )

    logging.info("Saving data")
    df.to_csv(f"{project_dir}/data/processed/{category}_diversity.csv", index=False)
