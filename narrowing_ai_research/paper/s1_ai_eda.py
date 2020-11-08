import numpy as np
import pandas as pd
import logging
import json
import pickle
import random
import altair as alt
import datetime
import yaml
from itertools import chain


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
    params = yaml.safe_load(infile)["section_1"]

# Functions
def load_process_data():
    """Loads AI paper data for analysis in section 1."""
    logging.info("Reading data")

    arxiv_cat_lookup = read_arxiv_cat_lookup()
    papers = read_papers()
    topic_long = read_topic_long()
    topic_mix = read_topic_mix()
    cats = read_arxiv_categories()

    logging.info("Reading tokenised abstracts")
    with open(f"{project_dir}/data/interim/arxiv_tokenised.json", "r") as infile:
        arxiv_tokenised = json.load(infile)

    logging.info("Reading AI labelling outputs")
    with open(f"{project_dir}/data/interim/find_ai_outputs.p", "rb") as infile:
        ai_indices, term_counts = pickle.load(infile)

    logging.info("Processing")
    papers["tokenised"] = papers["article_id"].map(arxiv_tokenised)

    # Create category sets to identify papers in different categories
    ai_cats = ["cs.AI", "cs.NE", "stat.ML", "cs.LG"]
    cat_sets = cats.groupby("category_id")["article_id"].apply(lambda x: set(x))

    # Create one hot encodings for AI categories
    ai_binary = pd.DataFrame(index=set(cats["article_id"]), columns=ai_cats)

    for c in ai_binary.columns:
        ai_binary[c] = [x in cat_sets[c] for x in ai_binary.index]

    # Create arxiv dataset
    papers.set_index("article_id", inplace=True)

    # We remove papers without abstracts and arXiv categories
    arx = pd.concat([ai_binary, papers], axis=1, sort=True).dropna(
        axis=0, subset=["abstract", "cs.AI"]
    )

    return arx, ai_indices, term_counts, arxiv_cat_lookup, cat_sets, cats, ai_cats


def make_agg_trend(arx, save=True):
    """Makes first plot"""
    # First chart: trends
    ai_bool_lookup = {False: "Other categories", True: "AI"}

    # Totals
    ai_trends = (
        arx.groupby(["date", "is_ai"]).size().reset_index(name="Number of papers")
    )
    ai_trends["is_ai"] = ai_trends["is_ai"].map(ai_bool_lookup)

    # Shares
    ai_shares = (
        ai_trends.pivot_table(index="date", columns="is_ai", values="Number of papers")
        .fillna(0)
        .reset_index(drop=False)
    )
    ai_shares["share"] = ai_shares["AI"] / ai_shares.sum(axis=1)

    #  Make chart
    at_ch = (
        alt.Chart(ai_trends)
        .transform_window(
            roll="mean(Number of papers)", frame=[-5, 5], groupby=["is_ai"]
        )
        .mark_line()
        .encode(
            x=alt.X("date:T", title="", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("roll:Q", title=["Number", "of papers"]),
            color=alt.Color("is_ai:N", title="Category"),
        )
        .properties(width=350, height=120)
    )
    as_ch = (
        alt.Chart(ai_shares)
        .transform_window(roll="mean(share)", frame=[-5, 5])
        .mark_line()
        .encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("roll:Q", title=["AI as share", "of all arXiv"]),
        )
    ).properties(width=350, height=120)

    ai_trends_chart = alt.vconcat(at_ch, as_ch, spacing=0)

    if save is True:
        save_altair(ai_trends_chart, "fig_1_ai_trends", driver=driv)

    return ai_trends_chart, ai_trends


def make_cumulative_results(trends, years):
    """Creates cumulative results"""
    cumulative = (
        trends.pivot_table(index="date", columns="is_ai", values="Number of papers")
        .fillna(0)
        .apply(lambda x: x / x.sum())
        .cumsum()
    )

    datetimes = [datetime.datetime(y, 1, 1) for y in years]

    paper_shares = cumulative.loc[
        [x.to_pydatetime() in datetimes for x in cumulative.index]
    ]

    return paper_shares


def make_category_distr_time(
    ai_indices, arx, cats, cat_sets, arxiv_cat_lookup, get_examples=False, example_n=4
):
    """Makes timecharts by category
    Args:
        ai_indices: dict where keys are categories and items paper ids in category
        arx: arxiv dataframe, used for the temporal analysis
        cats: lookup between categories and papers
        cat_sets: paper indices per category
        get_examples: whether we want to print examples
        example_n: number of examples to print
    """

    time_charts = []
    cat_charts = []

    logging.info("Trend analysis by category")
    # For each AI category and index
    for k, ind in ai_indices.items():

        logging.info(k)
        ind = set(ind)

        logging.info("Extracting category distribution")

        # Find all papers in the category
        # cat_subs = cats.loc[[x in ind for x in cats['article_id']]]
        cat_subs = cats.loc[cats["article_id"].isin(ind)]

        # Number of papers in category (without time)
        cat_distr = cat_subs["category_id"].value_counts().reset_index(name="n")

        cat_distr["category"] = k
        cat_charts.append(cat_distr)

        # Now get the year stuff
        rel_papers = arx.loc[ind]
        print(len(rel_papers))

        logging.info("Extracting trends")
        # Create timeline
        exp_year = rel_papers["date"].value_counts().reset_index(name="n")
        exp_year["category"] = k
        exp_year["type"] = "expanded"

        # Create timeline for core
        core_year = arx.loc[cat_sets[k]]["date"].value_counts().reset_index(name="n")
        core_year["category"] = k
        core_year["type"] = "core"

        # Combine
        combined_year = pd.concat([exp_year, core_year])
        time_charts.append(combined_year)

        linech = pd.concat(time_charts).reset_index(drop=True)

        linech["category_clean"] = linech["category"].map(arxiv_cat_lookup)

        # Show examples
        if get_examples is True:
            logging.info("Getting examples")
            rel_papers_ai = rel_papers["abstract"].tolist()
            picks = random.sample(rel_papers_ai, example_n)

            print("\n")
            for p in picks:
                print(p)
                print("\n")

    return linech, cat_charts


def make_cat_trend(linech, save=True, fig_n=2):
    """Makes chart 2"""

    ai_subtrends_chart = (
        alt.Chart(linech)
        .transform_window(
            roll="mean(n)", frame=[-10, 0], groupby=["category_clean", "type"]
        )
        .mark_line()
        .encode(
            x=alt.X("index:T", title=""),
            y=alt.X("roll:Q", title="Number of papers"),
            color=alt.Color("type", title="Source"),
        )
        .properties(width=200, height=100)
    ).facet(alt.Facet("category_clean", title="Category"), columns=2)

    if save is True:
        save_altair(ai_subtrends_chart, f"fig_{fig_n}_ai_subtrends", driver=driv)

    return ai_subtrends_chart


def make_cat_distr_chart(
    cat_sets, ai_joint, arxiv_cat_lookup, cats_to_plot=20, save=True, fig_n=3
):
    """Makes chart 3
    Args:
        cat_sets: df grouping ids by category
        ai_joint: ai ids
        arxiv_cat_lookup: lookup between arxiv cat ids and names
        cats_to_plot: Number of top categories to visualise
        save: Whether to save the plot

    """

    logging.info("Getting category frequencies")
    # Create a df with dummies for all categories
    cat_dums = []

    for c, v in cat_sets.items():
        d = pd.DataFrame(index=[x for x in v if x in ai_joint])
        d[c] = True
        cat_dums.append(d)

    cat_bin_df = pd.concat(cat_dums, axis=1, sort=True).fillna(0)

    # Category frequencies for all papers
    ai_freqs = (
        cat_bin_df.sum()
        .reset_index(name="paper_counts")
        .query("paper_counts > 0")
        .sort_values("paper_counts", ascending=False)
        .rename(columns={"index": "category_id"})
    )

    ai_freqs["category_name"] = [
        arxiv_cat_lookup[x][:40] + "..." for x in ai_freqs["category_id"]
    ]
    order_lookup = {cat: n for n, cat in enumerate(ai_freqs["category_id"])}
    ai_freqs["order"] = ai_freqs["category_id"].map(order_lookup)

    logging.info("Getting category overlaps")
    # Category frequencies for each category
    res = []
    for x in ai_freqs["category_id"]:

        ai_cat = cat_bin_df.loc[cat_bin_df[x] == True].sum().drop(x, axis=0)
        ai_cat.name = x
        res.append(ai_cat)

    hm_long = (
        pd.concat(res, axis=1)
        .apply(lambda x: x / x.sum(), axis=1)
        .loc[ai_freqs["category_id"]]
        .fillna(0)
        .reset_index(drop=False)
        .melt(id_vars="index")
    )

    hm_long["category_name_1"] = [
        arxiv_cat_lookup[x][:40] + "..." for x in hm_long["index"]
    ]
    hm_long["category_name_2"] = [
        arxiv_cat_lookup[x][:40] + "..." for x in hm_long["variable"]
    ]
    hm_long["order_1"] = hm_long["index"].map(order_lookup)
    hm_long["order_2"] = hm_long["variable"].map(order_lookup)
    hm_long["value"] = [100 * np.round(x, 4) for x in hm_long["value"]]

    # And plot
    logging.info("Plotting")
    # Barchart
    # We focus on the top 20 categories with AI papers
    ai_freq_bar = (
        alt.Chart(ai_freqs.loc[ai_freqs["order"] < cats_to_plot])
        .mark_bar(color="red", opacity=0.6, stroke="grey", strokeWidth=0.5)
        .encode(
            y=alt.Y("paper_counts", title="Number of papers"),
            x=alt.X(
                "category_name",
                title="",
                sort=alt.EncodingSortField("order"),
                axis=alt.Axis(labels=False, ticks=False),
            ),
        )
    ).properties(width=350, height=200)

    # HM
    ai_hm = (
        alt.Chart(hm_long.query("order_1 < 20").query("order_2 < 20"))
        .mark_rect()
        .encode(
            x=alt.X(
                "category_name_1",
                sort=alt.EncodingSortField("order_1"),
                title="arXiv category",
            ),
            y=alt.Y(
                "category_name_2",
                sort=alt.EncodingSortField("order_2"),
                title="arXiv category",
            ),
            # order=alt.Order('value',sort='ascending'),
            color=alt.Color(
                "value", title=["% of articles in x-category", "with y-category"]
            ),
            tooltip=["category_name_2", "value"],
        )
    ).properties(width=350)

    cat_freqs_hm = (
        alt.vconcat(ai_freq_bar, ai_hm)
        .configure_concat(spacing=0)
        .resolve_scale(color="independent")
    )

    if save is True:
        save_altair(cat_freqs_hm, f"fig_{fig_n}_arxiv_categories", driv)

    return cat_freqs_hm


def main():

    # Read data
    (
        arx,
        ai_indices,
        term_counts,
        arxiv_cat_lookup,
        cat_sets,
        cats,
        ai_cats,
    ) = load_process_data()

    # Extract results
    logging.info("AI descriptive statistics")
    results = {}

    # Q1: How many papers in total

    ai_expanded = set(chain(*[x for x in ai_indices.values()]))
    ai_core_dupes = list(chain(*[v for k, v in cat_sets.items() if k in ai_cats]))
    ai_core = set(chain(*[v for k, v in cat_sets.items() if k in ai_cats]))
    ai_new_expanded = ai_expanded - ai_core
    ai_joint = ai_core.union(ai_expanded)

    results["ai_expanded_n"] = len(ai_expanded)
    results["ai_core_with_duplicates_n"] = len(ai_core_dupes)
    results["ai_core_n"] = len(ai_core)
    results["ai_new_expanded_n"] = len(ai_new_expanded)
    results["ai_joint"] = len(ai_joint)

    # Plot chart 1:
    logging.info("Make first plot")
    plot_1, trends = make_agg_trend(arx)

    # Cumulative analysis
    # Get cumulative shares of activity
    logging.info("Cumulative results")

    year_cuml_shares = make_cumulative_results(trends, params["years"])

    # Add results
    for rid, r in year_cuml_shares.iterrows():
        results[f"Share of papers published before {str(rid.date())}"] = 100 * np.round(
            r["AI"], 2
        )

    # Get category timecharts
    timecharts, catcharts = make_category_distr_time(
        ai_indices, arx, cats, cat_sets, arxiv_cat_lookup, get_examples=False
    )

    make_cat_trend(timecharts)

    plot_3 = make_cat_distr_chart(cat_sets, ai_joint, arxiv_cat_lookup)

    with open(f"{project_dir}/reports/results.txt", "w") as outfile:
        for k, v in results.items():
            outfile.writelines(k + ": " + str(v) + "\n")


if __name__ == "__main__":
    driv = altair_visualisation_setup()
    main()
