import pandas as pd
import yaml
import logging
import altair as alt

from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_papers_orgs,
    read_topic_mix,
    query_orgs,
)
from narrowing_ai_research.paper.s3_org_eda import (
    paper_orgs_processing,
)
from narrowing_ai_research.utils.altair_utils import (
    altair_visualisation_setup,
    save_altair,
)

from narrowing_ai_research.transformers.diversity import Diversity, remove_zero_axis

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir


def read_process_data():
    papers = read_papers()
    paper_orgs = paper_orgs_processing(read_papers_orgs(), papers)
    paper_orgs["year"] = [x.year for x in paper_orgs["date"]]

    topic_mix = read_topic_mix()
    topic_mix.set_index("article_id", inplace=True)

    return papers, paper_orgs, topic_mix


def diversity_grid_search(data, metric_params, name, n_runs=None, sample_size=None):
    """Compare diversities between groups in the corpus
    Args:
        data: topic mix
        metric_params: dict where key is metric and values the par dicts
        name: name to label the outputs
        n_runs: if we want to run multiple runs (with samples)
        sample_size: size of sample for diversity based on random draws

    """
    logging.info(f"making diversity {name}")
    results = []
    # Here we remove axes with zero values
    data_ = remove_zero_axis(data)

    # For each metric, varams key value pair
    for d in metric_params.keys():
        logging.info(d)
        for n, b in enumerate(metric_params[d]):
            # If we only run once this extracts the result
            if n_runs is None:
                div = Diversity(data_, name)
                getattr(div, d)(b)
                r = [
                    name,
                    div.metric_name,
                    div.metric,
                    div.param_values,
                    f"Parametre Set {n}",
                ]
                results.append(r)
            # Multiple runs and appends all results
            else:
                b["sample"] = sample_size
                for r in range(n_runs):
                    if r % 10 == 0:
                        logging.info(n)
                    div = Diversity(data_, name)
                    getattr(div, d)(b)
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


def diversity_sector_comparison(
    data, topic_mix, metric_params, y0, n_runs=None, sample_size=None
):
    """Compares diversity of companies and academic institutions
    Args:
        data: paper - organisations
        topic_mix: df with topics
        metric_params: dict with key, values
        y0: earliest year to consider

    """
    logging.info("Subsetting data")
    # Subset to focus on recent years
    data_ = data.loc[data["year"] >= y0]

    # Extract ids for papers in companies and academia
    paper_ids_comp, paper_ids_academy = [
        query_orgs(data_, "org_type", t) for t in ["Company", "Education"]
    ]

    # Calculate diversities
    results_df = pd.concat(
        [
            diversity_grid_search(
                topic_mix.loc[ids], metric_params, name, n_runs, sample_size
            )
            for ids, name in zip(
                [paper_ids_comp, paper_ids_academy], ["Company", "Academic"]
            )
        ]
    )

    return results_df


def make_chart_sector_comparison(results_df, save=True, fig_num=12):
    """Barcharts comparing diversity of companies and academia"""

    div_comp_chart = (
        alt.Chart(results_df)
        .mark_bar(color="red", opacity=0.7, stroke="grey")
        .encode(
            x=alt.X("category", title="Category", sort=["Company", "Education"]),
            y=alt.Y("score", scale=alt.Scale(zero=False), title="Score"),
            column=alt.Column(
                "metric", title="Metric", sort=["balance", "weitzman", "rao_stirling"]
            ),
            row=alt.Row("parametre_set", title="Parametre set"),
        )
        .resolve_scale(y="independent")
    ).properties(height=75, width=100)

    if save is True:
        save_altair(div_comp_chart, f"fig_{fig_num}_div_sect", driv)

    return div_comp_chart


# Chart
def make_chart_sector_boxplot(df, save=True, fig_num=12):
    """Boxplots comparing diversity of companies and academia"""

    div_comp_sample = (
        alt.Chart(df)
        .mark_boxplot()
        .encode(
            x=alt.X("category:N", title="", sort=["Company", "Education"]),
            y=alt.Y("score:Q", scale=alt.Scale(zero=False)),
            row=alt.Row(
                "metric", title="Metric", sort=["balance", "weitzman", "rao-stirling"]
            ),
            column=alt.Column("parametre_set", title="Parametre set"),
            color=alt.Color("category", title="Organisation type"),
        )
        .resolve_scale(y="independent")
    ).properties(height=100, width=100)

    if save is True:
        save_altair(div_comp_sample, f"fig_{fig_num}_div_sect_multiple", driv)

    return div_comp_sample


def main():

    # We use the same diversity parametres as in the analysis of diversity
    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        pars = yaml.safe_load(infile)

    div_params = pars["section_4"]["div_params"]
    params = pars["section_7"]

    papers, paper_orgs, topic_mix = read_process_data()

    diversity_org_type_df = diversity_sector_comparison(
        paper_orgs, topic_mix, div_params, params["y0"]
    )

    bar_comp = make_chart_sector_comparison(diversity_org_type_df, save=True)

    diversity_org_type_multiple = diversity_sector_comparison(
        paper_orgs,
        topic_mix,
        div_params,
        y0=params["y0"],
        n_runs=params["runs"],
        sample_size=params["sample_size"],
    )

    box_comp = make_chart_sector_boxplot(diversity_org_type_multiple, save=True)


if __name__ == "__main__":
    alt.data_transformers.disable_max_rows()
    pd.options.mode.chained_assignment = None
    driv = altair_visualisation_setup()
    main()
