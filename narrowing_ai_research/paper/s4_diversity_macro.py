import pandas as pd
import numpy as np
import altair as alt
import logging
import networkx as nx
import yaml
from scipy.stats import zscore

from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_papers_orgs,
    read_topic_mix,
    read_topic_category_map,
    read_arxiv_cat_lookup,
)
from narrowing_ai_research.paper.s3_org_eda import (
    paper_orgs_processing,
)
from narrowing_ai_research.utils.altair_utils import (
    altair_visualisation_setup,
    save_altair,
)

from narrowing_ai_research.transformers.networks import (
    make_network_from_doc_term_matrix,
)
from narrowing_ai_research.transformers.diversity import Diversity

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir


def read_process_data():
    papers = read_papers()
    topic_mix = (
        remove_zero_axis(  # We remove a couple of papers with zero in all topics
            read_topic_mix().set_index("article_id")
        )
    )

    logging.info("Process dfs")
    papers["year"] = [x.year for x in papers["date"]]

    return (
        papers,
        topic_mix
    )


def normalise_diversity_year_df(y_div_df):
    """Normalises a dataframe with diversity information by year and parametre set"""
    yearly_results_norm = []

    # For each possible diversity metric it pivots over parametre sets
    # and calculates the zscore for the series
    for x in set(y_div_df["diversity_metric"]):
        yearly_long = y_div_df.query(f"diversity_metric == '{x}'").pivot_table(
            index=["year", "diversity_metric"], columns="parametre_set", values="score"
        )

        yearly_long_norm = yearly_long.apply(zscore)
        yearly_results_norm.append(yearly_long_norm)

    # Concatenate and melt so they can be visualised with altair
    y_div_df_norm = (
        pd.concat(yearly_results_norm)
        .reset_index(drop=False)
        .melt(
            id_vars=["year", "diversity_metric"],
            var_name="parametre_set",
            value_name="score",
        )
    )

    return y_div_df_norm


def year_diversity_results(data, year_ids, metric_params, t0=2008, t1=2020, norm=True):
    """Extracts yearly diversity results
    Args:
        data: topic mix
        metric_params: dict where keys are diversity metrics and
            values lists of parametres
        t0: first year
        t1: last year
        norm: if we want to obtain a normalised value (by year)
    """
    year_period = np.arange(t0, t1 + 1)

    year_results = []

    # Calculate diversity per year, metric and parametre set
    for y in year_period:
        logging.info(y)

        # Here we remove axes with zero values
        data_year = remove_zero_axis(data.loc[year_ids[y]])
        logging.info(data_year.shape)

        for d in metric_params.keys():
            for n, b in enumerate(metric_params[d]):
                div = Diversity(data_year, y)
                getattr(div, d)(b)
                r = [y, div.metric_name, div.metric, div.param_values, f"param_set_{n}"]
                year_results.append(r)

    yearly_results_df = pd.DataFrame(
        year_results,
        columns=[
            "year",
            "diversity_metric",
            "score",
            "parametres_detailed",
            "parametre_set",
        ],
    )

    if norm is True:
        yearly_results_norm_df = normalise_diversity_year_df(yearly_results_df)
        return yearly_results_norm_df

    else:
        return yearly_results_df


def make_chart_diversity_evol(results, save=True, fig_n=10):
    """Plots evolution of diversity"""
    div_evol_ch = (
        (
            alt.Chart(results)
            .mark_line(opacity=0.9)
            .encode(
                x=alt.X("year:O", title=""),
                y=alt.Y("score:Q", scale=alt.Scale(zero=False), title="z-score"),
                row=alt.Row(
                    "diversity_metric",
                    title="Diversity metric",
                    sort=["balance", "weitzman", "rao_stirling"],
                ),
                color=alt.Color("parametre_set:N", title="Parameter set"),
            )
        )
        .resolve_scale(y="independent")
        .properties(width=250, height=80)
    )

    if save is True:
        save_altair(div_evol_ch, f"fig_{fig_n}_div_evol", driv)

    return div_evol_ch


def extract_distribution_centrality(
    data,
    year_ids,
    threshold=0.05,
    ranking_distr=[20, 50, 100, 200],
    ranking_centrality=[20, 50, 100, 200, 400, 600],
    t0=2008,
    t1=2020,
):
    """Extracts topic distribution and centrality under various thresholds
    Args:
        data: topic mix
        year_ids: ids for the years
        threshold: threshold for inclusion of topics
        ranking_distr: rankings to estimate share of distribution accounted
        ranking_centrality: rankings to compare evolution of centrality

    Returns shares and eigenvector centralities for topics in different positions
    of the activity distribution (by shares of activity)
    """
    # Balance:
    # What are the % of activity accounted for the most popular topics?
    act_top = []
    centrality_ranked = []

    # Extract the share of activity accountes by top 10,25,50,100 topics each year
    for y in np.arange(t0, t1 + 1):
        logging.info(y)

        # Activity in year
        y_t = data.loc[data.index.isin(year_ids[y])]
        y_t = remove_zero_axis(y_t)

        # Total activity by topic in the year
        topic_distr = y_t.idxmax(axis=1).value_counts(normalize=True).cumsum()

        # Topic co-occurrence network
        net = make_network_from_doc_term_matrix(
            y_t.reset_index(drop=False), id_var="article_id", threshold=0.05
        )

        # Extract eigenvector centrality from the network
        ev = pd.Series(nx.eigenvector_centrality(net))

        # These are sorted by level of activity (as in topic distr above)
        index_for_sort = [x for x in topic_distr.index if x in ev.index]

        ev_df = (
            ev.loc[index_for_sort]
            .reset_index(drop=False)
            .rename(columns={"index": "topic", 0: "eigenvector_centrality"})
        )

        ev_df["eigenvector_z"] = zscore(ev_df["eigenvector_centrality"])
        ev_df["rank"] = range(0, len(ev_df))
        ev_df["year"] = y

        shares = []

        for n in ranking_distr:
            shares.append(topic_distr[n])

        act_top.append(pd.Series(shares, name=y, index=[x for x in ranking_distr]))
        centrality_ranked.append(ev_df)

    shares_long = (
        pd.concat(act_top, axis=1).reset_index(drop=False).melt(id_vars="index")
    )

    centrality_ranked_all = pd.concat(centrality_ranked)
    centrality_ranked_all["rank_segment"] = centrality_ranked_all["rank"].apply(
        allocate_rank, thres=ranking_centrality
    )

    return shares_long, centrality_ranked_all


def make_chart_distribution_centrality(
    shares_long, centrality_ranked_all, saving=True, fig_n=11
):
    """Plots evolution of shares of topics and centrality averages
    at different positions of the distribution
    Args:
        shares_long: df with shares of topic activity accounted at
            different points of the distribution
        centrality_ranked_all: df with mean centralities for topics
            at different points of the distribution

    """

    shares_evol = (
        alt.Chart(shares_long)
        .mark_line()
        .encode(
            x=alt.X("variable:N", title="", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("value", title=["Share of activity", "accounted by rank"]),
            color=alt.X("index:N", title="Position in distribution"),
        )
    ).properties(width=300, height=170)

    line = (
        alt.Chart(centrality_ranked_all)
        .transform_aggregate(m="mean(eigenvector_z)", groupby=["year", "rank_segment"])
        .mark_line()
        .encode(x=alt.X("year:N", title=""), y="m:Q", color="rank_segment:N")
    )

    band = (
        alt.Chart(centrality_ranked_all)
        .mark_errorband()
        .encode(
            x="year:N",
            y=alt.Y("eigenvector_z", title=["Mean eigenvector", "centrality"]),
            color=alt.Color("rank_segment:N", title="Position in distribution"),
        )
    )

    eigen_evol_linech = (band + line).properties(width=300, height=170)

    div_comp = alt.vconcat(shares_evol, eigen_evol_linech, spacing=0).resolve_scale(
        x="shared", color="independent"
    )

    if saving is True:
        save_altair(div_comp, f"fig_{fig_n}_div_comps_evol", driv)

    return div_comp


def allocate_rank(x, thres):
    """Allocates a value to its rank based on thresholds"""
    out = np.nan

    for v in thres:

        if x < v:
            return v

    return out


def remove_zero_axis(df):
    """Removes axes where all values are zero"""

    df_r = df.loc[df.sum(axis=1) > 0, df.sum() > 0]
    return df_r


def main():
    logging.info("Reading data")

    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        div_params = yaml.safe_load(infile)["section_4"]["div_params"]

    papers, topic_mix = read_process_data()

    year_ids = (
        papers.loc[papers["is_ai"] is True]
        .groupby("year")["article_id"]
        .apply(lambda x: set(x))
    )

    logging.info("Calculate yearly diversity")
    yearly_diversity_norm = year_diversity_results(topic_mix, year_ids, div_params)

    div_year_ch = make_chart_diversity_evol(yearly_diversity_norm, save=True)

    logging.info("Evolution of concentration and centrality")
    shares_long, centrality_ranked_all = extract_distribution_centrality(
        topic_mix, year_ids
    )

    conc_centr_ch = make_chart_distribution_centrality(
        shares_long, centrality_ranked_all, saving=True
    )


if __name__ == "__main__":
    alt.data_transformers.disable_max_rows()
    pd.options.mode.chained_assignment = None
    driv = altair_visualisation_setup()
    main()
