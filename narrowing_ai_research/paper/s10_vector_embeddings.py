import pandas as pd
import logging
import yaml
import altair as alt
from sklearn.manifold import TSNE
from narrowing_ai_research.utils.altair_utils import (
    save_altair,
    altair_visualisation_setup,
)

from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_papers_orgs,
    read_topic_mix,
    paper_orgs_processing,
    read_vectors,
)

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir


def read_process_data():
    papers = read_papers()
    papers_orgs = paper_orgs_processing(read_papers_orgs(), papers)
    topic_mix = read_topic_mix()
    topic_mix.set_index("article_id", inplace=True)
    vectors = read_vectors().pivot_table(
        index='article_id',columns='dimension',values='value'
    )

    return papers, papers_orgs, topic_mix, vectors


def google_process(papers_orgs):
    """
    Separates Google and DeepMind papers
    """
    # Finds google papers
    google_papers = papers_orgs.loc[papers_orgs.org_name == "Google"]

    dm_ids = set(papers_orgs.loc[papers_orgs["org_name"] == "DeepMind"]["article_id"])

    # Removes any with deepmind ids
    google_subs = google_papers.loc[~google_papers["article_id"].isin(dm_ids)]

    # Concats with a df removing papers overlapping between google and deepmind
    return pd.concat(
        [papers_orgs.loc[papers_orgs["org_name"] != "Google"], google_subs]
    ).reset_index(drop=True)


def make_combined_tsne_df(
    activity_thresholds, period, vectors_wide, papers, papers_orgs, highlight_orgs
):
    """Creates TSNE dfs for different levels of activity
    Args:
        activity_thresholds: list of thresholds to select org groups
            with varying size / ;evel of importance
        period: period to consider
        vectors_wide: wide df with abstract embeddings
        papers: paper metadata
        paper_orgs: papers with org data
        highlight_orgs: how many orgs to highlight
    """

    # Org name and type lookup
    org_name_type_lookup = (
        papers_orgs.drop_duplicates("org_name")
        .set_index("org_name")["org_type"]
        .to_dict()
    )

    ids_ = set(
        papers.loc[(papers["is_ai"] == True) & (papers["year"].isin(period))][
            "article_id"
        ]
    )

    # Create a list with names of organisations in various rankings
    org_list = [
        papers_orgs.loc[papers_orgs["article_id"].isin(ids_)]["org_name"]
        .value_counts()
        .head(n=v)
        .index
        for v in activity_thresholds
    ]

    vector_comparison = [
        make_tsne_df(
            orgs, period, vectors_wide, papers, papers_orgs, org_name_type_lookup
        )
        for orgs in org_list
    ]

    tsne_combi = pd.concat(
        [x[1].assign(size=n) for x, n in zip(vector_comparison, activity_thresholds)]
    )

    # Labels top X organisations
    tsne_combi["top"] = [
        x in org_list[0].tolist()[:highlight_orgs] for x in tsne_combi["index"]
    ]

    tsne_df = tsne_combi.dropna(axis=0, subset=["org_type"])

    return tsne_df

def make_tsne_df(orgs, period, vectors, papers, papers_orgs, org_name_type_lookup):
    """Returns a TSNE df ready to visualise"""

    # Extract ids for papers in period
    logging.info("Subsetting data")
    period_ids = set(
        papers.loc[papers["is_ai"] == True].loc[papers["year"].isin(period)][
            "article_id"
        ]
    )

    # Subset the vector with these
    period_vectors = vectors.loc[vectors.index.isin(period_ids)]

    # Find papers in the year and organisation
    orgs_paper_sets = (
        papers_orgs.loc[
            (papers_orgs["article_id"].isin(period_ids))
            & (papers_orgs["org_name"].isin(orgs))
        ]
        .groupby("org_name")["article_id"]
        .apply(lambda x: set(x))
    )

    # Get an org_activity_lookup
    org_activity_lookup = {k: len(v) for k, v in orgs_paper_sets.items()}

    org_vectors = []

    # Note that this is already focusing on papers in the priod
    # (we subset that above)
    for org in orgs:
        # We ignore organisations not present in the data in the period
        if org in orgs_paper_sets.keys():
            sel_org_vect = period_vectors.loc[
                period_vectors.index.isin(orgs_paper_sets[org])
            ].mean()
            sel_org_vect.name = org
            org_vectors.append(sel_org_vect)

    # Create a df. We remove columns (organisations) with missing values
    org_vector_df = pd.concat(org_vectors, axis=1).dropna(axis=1)

    logging.info("Fitting TSNE")
    # TSNE
    tsne = TSNE()
    tsne_vect = tsne.fit_transform(org_vector_df.T)

    # Process the output
    logging.info("Outputting data")
    tsne_df = pd.DataFrame(tsne_vect, index=org_vector_df.columns, columns=["x", "y"])

    tsne_df["org_type"] = tsne_df.index.map(org_name_type_lookup)
    tsne_df["activity"] = tsne_df.index.map(org_activity_lookup)
    tsne_df["period"] = "-".join([str(period[0]), str(period[-1])])

    tsne_df = tsne_df.reset_index(drop=False)

    logging.info("Saving data")
    return org_vector_df, tsne_df


def visualise_tsne(tsne_df, save=True, fig_num=15):
    """Visualise tsne plot"""
    tsne_base = alt.Chart(tsne_df).encode(
        x=alt.X("x:Q", title="", axis=alt.Axis(ticks=False, labels=False)),
        y=alt.Y("y:Q", title="", axis=alt.Axis(ticks=False, labels=False)),
    )

    tsne_points = (
        (
            tsne_base.mark_point(
                filled=True, opacity=0.5, stroke="black", strokeOpacity=0.5
            ).encode(
                color=alt.Color("org_type", title="Organisation type"),
                strokeWidth=alt.Stroke(
                    "top", scale=alt.Scale(range=[0, 1]), legend=None
                ),
                # stroke = alt.value('blue'),
                size=alt.Size("activity:Q", title="Number of papers"),
                facet=alt.Facet(
                    "size", columns=2, title="Number of organisations in plot"
                ),
                tooltip=["index"],
            )
        )
        .interactive()
        .resolve_scale(y="independent", x="independent")
        .properties(width=250, height=250)
    )

    if save is True:
        save_altair(tsne_points, "fig_15_tsne", driv)

    return tsne_points


def main():
    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        pars = yaml.safe_load(infile)["section_10"]

    papers, papers_orgs, topic_mix, vectors = read_process_data()

    logging.info("Process google papers")
    papers_orgs_ = google_process(papers_orgs)

    logging.info("Making TSNE df")
    combi_df = make_combined_tsne_df(
        pars["activity_thresholds"],
        pars["period"],
        vectors,
        papers,
        papers_orgs_,
        pars["highlights_n"],
    )

    logging.info("Visualising Tsne")
    tsne = visualise_tsne(combi_df, save=True)

if __name__ == "__main__":
    driv = altair_visualisation_setup()
    alt.data_transformers.disable_max_rows()
    pd.options.mode.chained_assignment = None
    main()
