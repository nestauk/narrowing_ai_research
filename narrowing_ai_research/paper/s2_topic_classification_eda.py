import numpy as np
import pandas as pd
import logging
import json
import random
import altair as alt
import yaml
import re

import narrowing_ai_research
from narrowing_ai_research.utils.altair_utils import (
    altair_visualisation_setup,
    save_altair,
)
from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_topic_mix,
    read_topic_long,
    read_arxiv_cat_lookup,
    read_arxiv_categories,
)

project_dir = narrowing_ai_research.project_dir

with open(f"{project_dir}/paper_config.yaml", "r") as infile:
    params = yaml.safe_load(infile)["section_2"]


def read_process_data():
    """Reads and processes the data"""
    arxiv_cat_lookup = read_arxiv_cat_lookup()
    papers = read_papers()
    topic_long = read_topic_long()
    topic_mix = read_topic_mix()
    cats = read_arxiv_categories()

    # Paper cats
    cat_sets = cats.groupby(["category_id"])["article_id"].apply(lambda x: set(x))

    # create a unique cat_sets
    one_cat_ps = cats.groupby("article_id")["category_id"].apply(lambda x: len(x))
    one_cat_ids = set(one_cat_ps.loc[one_cat_ps == 1].index)

    return papers, topic_mix, topic_long, cats, cat_sets, one_cat_ids, arxiv_cat_lookup


def make_topic_cat_specialisation(
    topic_mix,
    cat_sets,
    one_cat_ids,
    top_n_categories=20,
    topic_threshold=0.1,
    unique=True,
):
    """Measure topic specialisation in categories
    Args:
        topic_mix: topic mix with topic weights and probabilities
        cat_set: index of papers by category
        one_cat_ids: ids for papers in a single category
        top_n_categories: number of arXiv categories to consider
        topic_threshold: threshold for considering that a topic is in a paper
        unique: we only consider papers with a single category
    """

    topic_mix_ = topic_mix.copy()

    # Extract topic counts per category and top categories in AI
    logging.info("Identifying top categories")

    ai_ids = set(topic_mix_["article_id"])

    ai_papers = []
    for c in cat_sets.keys():
        if unique is True:
            ai_p = len(ai_ids & cat_sets[c] & one_cat_ids)
        else:
            ai_p = len(ai_ids & cat_sets[c])
        ai_papers.append([c, ai_p])

    # Counts of AI papers in different categories
    ai_counts_df = pd.DataFrame(ai_papers, columns=["category", "count"])

    # Identify categories with more AI appers
    top_ai_cats = set(
        ai_counts_df.sort_values("count", ascending=False).head(n=top_n_categories)[
            "category"
        ]
    )

    logging.info(", ".join(list(top_ai_cats)))

    logging.info("Calculating topic specialisation")
    all_cat_topic_rca = []

    # If we only want to focus on papers with a single categoru
    if unique is True:
        topic_mix_ = topic_mix_.loc[topic_mix_["article_id"].isin(one_cat_ids)]

    # Share of topic in the population
    topic_mix_shares = (
        topic_mix_.iloc[:, 1:].applymap(lambda x: x > topic_threshold).mean()
    )

    # For each topic we calculate the specialisation
    for cat in top_ai_cats:
        logging.info(cat)

        cat_ids = cat_sets[cat]

        # Find papers in category
        cat_topics = topic_mix_.loc[topic_mix_["article_id"].isin(cat_ids)].set_index(
            "article_id"
        )

        # Binarise and calculate shares
        cat_topics_bin = cat_topics.applymap(lambda x: x > topic_threshold).mean()

        # RCA
        cat_topics_rca = cat_topics_bin / topic_mix_shares

        cat_topics_rca.name = cat

        all_cat_topic_rca.append(cat_topics_rca)

    return pd.DataFrame(all_cat_topic_rca)


def extract_topics(topic_rca, spec_thres):
    """Creates a map between topics and their core categories
    Args:
        topic_rca: specialisation topic
        spec_thres: threshold to assess specialisation

    """
    # Identify specialised topics
    spec_topics = topic_rca.max().mask(topic_rca.max() < spec_thres).dropna().index

    # Find the max category for each topic
    topic_category_allocation_ = topic_rca.idxmax().to_dict()

    # Create the dictionary
    topic_category_allocation = {
        k: v for k, v in topic_category_allocation_.items() if k in spec_topics
    }

    return topic_category_allocation


def extract_topic_trends(
    topic_long,
    topic_category_allocation,
    period=[2005, 2021],
    window=10,
    topic_value=0.1,
    shares=True,
):
    """Creates a df with levels of activity per topic and year
    Args:
        topic_long: df with topic activity per year
        topic_category_allocation:
                    maps topics vs the categories where they are most salient
        period: period to consider
        window: window for rolling averages
        topic_value: threshold value for topic
        shares: report normalised

    """
    logging.info("Calculating trends")
    # Label topics with categories
    topic_long_ = topic_long.copy()

    topic_long_["topic_cat"] = topic_long_["variable"].map(topic_category_allocation)

    # Calculate trends
    topic_trends = (
        topic_long_.query(f"value > {topic_value}")
        .groupby(["date", "topic_cat"])
        .size()
        .reset_index(name="value")
    )

    # Focus on period of interest
    topic_trends = topic_trends.loc[
        [x.year in np.arange(period[0], period[1]) for x in topic_trends["date"]]
    ]

    # If shares, normalise
    if shares is True:
        logging.info("Normalising")
        topic_trends = topic_trends.pivot_table(
            index="date", columns="topic_cat", values="value"
        )
        topic_trends = (
            topic_trends.apply(lambda x: x / x.sum(), axis=1)
            .reset_index(drop=False)
            .melt(id_vars="date")
        )

    logging.info("Rolling means")
    topic_trends_rolling = (
        topic_trends.pivot_table(index="date", columns="topic_cat", values="value")
        .fillna(0)
        .rolling(window=window)
        .mean()
        .dropna()
        .reset_index(drop=False)
        .melt(id_vars="date")
    )

    return topic_trends_rolling


def micro_trends(papers, topic_mix, topic_list, threshold, name):
    """Returns a table with trends by topic
    Args:
        papers: paper metadata
        topoc_mix: wide version of the topic mix
        topic_list: topics we want to focus on
        threshold: threshold for including a variable
        name: names for the topic groups
    """
    # Number of papers per date
    paper_trends = papers.loc[papers["is_ai"] == True].groupby("date").size()

    # Identify papers in topic (if any of the topics is above the threshold)
    paper_in_topic = set(
        [
            row["article_id"]
            for _id, row in topic_mix.iterrows()
            if any(row[t] > threshold for t in topic_list)
        ]
    )

    # Count papers in topic by date
    topic_trends = (
        papers.loc[papers["article_id"].isin(paper_in_topic)].groupby("date").size()
    )

    # Concatenate with the overall levels of activity and normalise
    out = pd.concat([paper_trends, topic_trends], axis=1).fillna(0)
    out.columns = ["all_papers", name]
    out[f"{name}_share"] = out[name] / out["all_papers"]

    return out


def make_chart_topic_spec(
    topic_rca,
    topic_mix,
    arxiv_cat_lookup,
    topic_thres=0.05,
    topic_n=150,
    save=False,
    fig_n="extra_1",
):
    """Visualises prevalence of topics in a category
    Args:
        topic_rca: relative specialisation of topics in categories
        arxiv_cat_lookup: lookup between category ids and names
        topic_thres: threshold for topic
        topic_n: number of topics to consider
        save: if we want to save the figure
        fig_n: figure id

    """
    logging.info("Extracting topic counts")
    # Visualise topic distributions
    topic_counts_long = topic_rca.reset_index(drop=False).melt(id_vars="index")

    # Extract top topics
    top_topics = list(
        topic_mix.iloc[:, 1:]
        .applymap(lambda x: x > topic_thres)
        .sum(axis=0)
        .sort_values(ascending=False)[:topic_n]
        .index
    )

    # Focus on those for the long topic
    topic_counts_long_ = topic_counts_long.loc[
        topic_counts_long["variable"].isin(top_topics)
    ]

    # Add nice names for categoru
    topic_counts_long_["arx_cat"] = [
        x.split(" ") for x in topic_counts_long_["index"].map(arxiv_cat_lookup)
    ]

    topic_spec = (
        alt.Chart(topic_counts_long_)
        .mark_bar(color="red")
        .encode(
            y=alt.Y(
                "variable", sort=top_topics, axis=alt.Axis(labels=False, ticks=False)
            ),
            x="value",
            facet=alt.Facet("arx_cat", columns=5),
            tooltip=["variable", "value"],
        )
    ).properties(width=100, height=100)

    if save is True:
        save_altair(topic_spec, f"fig_{fig_n}_topic_specialisations", driv)

    return topic_spec


def make_chart_topic_trends(
    topic_trends, arxiv_cat_lookup, year_sort=2020, save=True, fig_n=4
):
    """Topic trend chart"""

    # Sort topics by the year of interest
    topics_sorted = (
        topic_trends.loc[[x.year == year_sort for x in topic_trends["date"]]]
        .groupby("topic_cat")["value"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    topic_trends["order"] = [
        [n for n, k in enumerate(topics_sorted) if x == k][0]
        for x in topic_trends["topic_cat"]
    ]

    # Create clean category names
    topic_trends["topic_cat_clean"] = [
        arxiv_cat_lookup[x][:50] + "..." for x in topic_trends["topic_cat"]
    ]

    # Create clean topic sorted names
    topics_sorted_2 = [arxiv_cat_lookup[x][:50] + "..." for x in topics_sorted]

    evol_sh = (
        alt.Chart(topic_trends)
        .mark_bar(stroke="grey", strokeWidth=0.1)
        .encode(
            x="date:T",
            y=alt.Y("value", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "topic_cat_clean",
                sort=topics_sorted_2,
                title="Source category",
                scale=alt.Scale(scheme="tableau20"),
                legend=alt.Legend(columns=2),
            ),
            order=alt.Order("order", sort="descending"),
            tooltip=["topic_cat_clean"],
        )
    ).properties(width=400)

    if save is True:
        save_altair(evol_sh, f"fig_{fig_n}_topic_trends", driv)

    return evol_sh


def plot_microtrends(
    trend_table, t0=2010, t1=2021, window=10, save=False, name="microtrends_example"
):
    """Plot microtrends:
    Args:
        trend_table: table with trends in a topic
        t0, t1: first and last year
        window: window for rolling averages
        save: if we want to save
        name: name (only needed if we want to save)
    """

    ex_trends_norm = (
        trend_table.loc[[x.year in np.arange(t0, t1) for x in trend_table.index]]
        .rolling(window=10)
        .mean()
        .dropna()
        .reset_index(drop=False)
        .melt(id_vars="date")
    )

    tr = (
        alt.Chart(ex_trends_norm)
        .mark_line()
        .encode(x="date:T", y="value", color="variable")
    )

    if save is True:
        save_altair(tr, f"fig_{name}", driv)

    return tr


def make_cat_topic_table(
    topic_category_map,
    cats=[
        "cs.AI",
        "cs.NE",
        "cs.LG",
        "stat.ML",
        "cs.CV",
        "cs.CL",
        "cs.CR",
        "cs.IR",
        "cs.CY",
    ],
):
    """Show examples of topics in different categories"""

    topics_table = (
        pd.Series(topic_category_map)
        .reset_index(drop=False)
        .groupby(0)["index"]
        .apply(
            lambda x: [
                re.sub("_", "\_", t) for t in topic_category_map.keys() if t in list(x)
            ]
        )
        .loc[cats]
    )
    with open(f"{project_dir}/reports/tables/topics_table.txt", "w") as outfile:
        for k in topics_table.keys():
            outfile.writelines(
                k
                + " & "
                + r"\newline ".join(
                    [x[:80] + "... " for x in random.sample(topics_table[k][:15], 4)]
                )
            )
            outfile.writelines("\n")
            outfile.writelines(r"\\ \\")
            outfile.writelines("\n")


def main():

    (
        papers,
        topic_mix,
        topic_long,
        cats,
        cat_sets,
        one_cat_ids,
        arxiv_cat_lookup,
    ) = read_process_data()

    logging.info("Calculating topic specialisation")
    topic_rca = make_topic_cat_specialisation(
        topic_mix, cat_sets, one_cat_ids, topic_threshold=params["topic_thres"]
    )

    topic_spec_chart = make_chart_topic_spec(
        topic_rca,
        topic_mix,
        arxiv_cat_lookup,
        save=True,
    )
    topic_category_map = extract_topics(topic_rca, spec_thres=params["rca_thres"])

    logging.info("Visualising topic trends")
    topic_trends = extract_topic_trends(
        topic_long,
        topic_category_map,
        topic_value=params["trends_topic_value"],
        window=15,
    )

    topic_trend = make_chart_topic_trends(
        topic_trends, arxiv_cat_lookup, save=True, year_sort=2006
    )

    logging.info("Saving outputs")
    with open(f"{project_dir}/data/interim/topic_category_map.json", "w") as outfile:
        json.dump(topic_category_map, outfile)

    make_cat_topic_table(topic_category_map)


if __name__ == "__main__":
    driv = altair_visualisation_setup()
    main()
