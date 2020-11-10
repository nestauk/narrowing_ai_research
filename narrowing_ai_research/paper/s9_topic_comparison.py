import pandas as pd
import numpy as np
import altair as alt
import yaml
import re
import logging
from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_papers_orgs,
    read_topic_mix,
    read_arxiv_cat_lookup,
    read_topic_category_map,
    query_orgs,
    paper_orgs_processing,
)
from narrowing_ai_research.utils.altair_utils import (
    save_altair,
    altair_visualisation_setup,
)

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir


def read_process_data():
    papers = read_papers()
    paper_orgs = paper_orgs_processing(read_papers_orgs(), papers)
    topic_mix = read_topic_mix()
    topic_mix.set_index("article_id", inplace=True)

    # topic_long = read_topic_long()
    topic_category_map = read_topic_category_map()
    arxiv_cat_lookup = read_arxiv_cat_lookup()
    topic_list = topic_mix.columns

    return (
        papers,
        paper_orgs,
        topic_mix,
        topic_category_map,
        arxiv_cat_lookup,
        topic_list,
    )


def topic_rep(ids, topic_mix, my_cats, topic_list, topic_category_map, thres=0.05):
    """Compares representation of topics in a category with the alternative
    Args:
        ids: papers by orgs in the category we are interested in
        topic_mix: topic mix
        my_cats: categories we are instested in
        threshold: threshold for considering a paper has a topic
    """
    topic_mix_ = topic_mix.copy().assign(rel=lambda x: x.index.isin(ids))

    # Calculates share of papers in category with the rest
    topic_presences = (
        topic_mix_[topic_list]
        .applymap(lambda x: x > thres)
        .groupby(topic_mix_["rel"])[topic_list]
        .mean()
    )

    topic_comparison = (
        ((topic_presences.loc[True] / topic_presences.loc[False]) - 1)
        .sort_values(ascending=False)
        .reset_index(name="ratio")
    )

    topic_comparison["cat"] = topic_comparison["index"].map(topic_category_map)

    # Selects categories
    topic_comparison["cat_sel"] = [
        x if x in my_cats else np.nan for x in topic_comparison["cat"]
    ]

    return topic_comparison, topic_presences


def make_chart_topic_comparison(
    topic_mix,
    arxiv_cat_lookup,
    comparison_ids,
    selected_categories,
    comparison_names,
    topic_list,
    topic_category_map,
    highlights=False,
    highlight_topics=None,
    highlight_class_table="Company",
    save=True,
    fig_num=15,
):
    """Creates a chart that compares the topic specialisations
    of different groups of organisations
    Args:
        topic_mix: topic mix
        arxiv_cat_lookup: lookup between category ids and names
        comparison_ids: ids we want to compare
        selected_categories: arXiv categories to focus on
        comparison_names: names for the categories we are comparing
        highlights: if we want to highlight particular topics
        highlight_topics: which ones
        highlight_class_table: topics to highlight in the table
    """

    # Extract the representations of categories
    comp_topic_rel = pd.DataFrame(
        [
            topic_rep(
                ids,
                topic_mix,
                selected_categories,
                topic_list=topic_list,
                topic_category_map=topic_category_map,
            )[1].loc[True]
            for ids in comparison_ids
        ]
    )
    comparison_df = comp_topic_rel.T
    comparison_df.columns = comparison_names

    comparison_df_long = comparison_df.reset_index(drop=False).melt(id_vars="index")
    comparison_df_long["cat"] = comparison_df_long["index"].map(topic_category_map)

    order = (
        comparison_df_long.groupby(["index", "cat"])["value"]
        .sum()
        .reset_index(drop=False)
        .sort_values(by=["cat", "value"], ascending=[True, False])["index"]
        .tolist()
    )

    comparison_df_filter = comparison_df_long.loc[
        comparison_df_long["cat"].isin(selected_categories)
    ]

    comparison_df_filter["cat_clean"] = [
        arxiv_cat_lookup[x][:35] + "..." for x in comparison_df_filter["cat"]
    ]

    # Sort categories by biggest differences?
    diff_comp = (
        comparison_df_filter.pivot_table(
            index=["index", "cat_clean"], columns="variable", values="value"
        )
        .assign(diff=lambda x: x["company"] - x["academia"])
        .reset_index(drop=False)
        .groupby("cat_clean")["diff"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Plot
    comp_ch = (
        alt.Chart(comparison_df_filter)
        .mark_point(filled=True, opacity=0.5, stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "index", title="", sort=order, axis=alt.Axis(labels=False, ticks=False)
            ),
            y=alt.Y("value", title=["Share of papers", "with topic"]),
            color=alt.Color("variable", title="Organisation type"),
            tooltip=["index"],
        )
    )

    comp_lines = (
        alt.Chart(comparison_df_filter)
        .mark_line(strokeWidth=1, strokeDash=[1, 1], stroke="grey")
        .encode(
            x=alt.X("index", sort=order, axis=alt.Axis(labels=False, ticks=False)),
            y="value",
            detail="index",
        )
    )

    topic_comp_type = (
        (comp_ch + comp_lines)
        .properties(width=200, height=150)
        .facet(
            alt.Facet("cat_clean", sort=diff_comp, title="arXiv category"), columns=3
        )
        .resolve_scale(x="independent")
    )

    if highlights is False:

        topic_comp_type = (
            (comp_ch + comp_lines)
            .properties(width=200, height=150)
            .facet(
                alt.Facet("cat_clean", sort=diff_comp, title="arXiv category"),
                columns=3,
            )
            .resolve_scale(x="independent")
        )

        if save is True:
            save_altair(topic_comp_type, f"fig_{fig_num}_topic_comp", driv)

        return topic_comp_type
    else:

        # Lookup for the selected categories
        code_topic_lookup = {v: str(n + 1) for n, v in enumerate(highlight_topics)}

        # Add a label per topic for the selected topics
        comparison_df_filter["code"] = [
            code_topic_lookup[x] if x in code_topic_lookup.keys() else "no_label"
            for x in comparison_df_filter["index"]
        ]

        # Need to find a way to remove the bottom one
        max_val = comparison_df_filter.groupby("index")["value"].max().to_dict()
        comparison_df_filter["max"] = comparison_df_filter["index"].map(max_val)

        comp_text = (
            alt.Chart(comparison_df_filter)
            .transform_filter(alt.datum.code != "no_label")
            .mark_text(yOffset=-10, color="red")
            .encode(
                text=alt.Text("code"),
                x=alt.X("index", sort=order, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("max", title=""),
                detail="index",
            )
        )

        topic_comp_type = (
            (comp_ch + comp_lines + comp_text)
            .properties(width=200, height=150)
            .facet(
                alt.Facet("cat_clean", sort=diff_comp, title="arXiv category"),
                columns=3,
            )
            .resolve_scale(x="independent")
        )

        if save is True:
            save_altair(topic_comp_type, "fig_9_topic_comp", driv)
            save_highlights_table(
                comparison_df_filter,
                highlight_topics,
                highlight_class_table,
                topic_category_map,
            )

        return topic_comp_type, comparison_df_filter


def save_highlights_table(
    comparison_table, labels_to_display, highlight_label, topic_category_map
):
    """Saves a latex table with comparison labels"""
    # Create a latex table for the paper
    comparison_pivot = comparison_table.pivot_table(
        index="index", columns="variable", values="value"
    )
    comparison_pivot["win"] = comparison_pivot.idxmax(axis=1)
    comparison_pivot_lookup = comparison_pivot["win"].to_dict()

    with open(f"{project_dir}/reports/tables/comparison_table.txt", "w") as outfile:

        for n, x in enumerate(labels_to_display):

            x2 = re.sub(r"_", " ", x)

            comp = comparison_pivot_lookup[x].capitalize()
            if comp is highlight_label:
                t = "\\textbf{" + comp + "}"
            else:
                t = comp

            row = f"{str(n+1)} & {x2} & {topic_category_map[x]} & {t}"
            outfile.writelines(row)
            outfile.writelines("\n")
            outfile.writelines(r"\\ \\")
            outfile.writelines("\n")


def main():
    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        pars = yaml.safe_load(infile)["section_9"]

    cats = pars["categories"]
    labels_to_display = pars["topic_highlights"]

    (
        papers,
        porgs,
        topic_mix,
        topic_category_map,
        arxiv_cat_lookup,
        topic_list,
    ) = read_process_data()

    comp_ids, acad_ids = [
        query_orgs(porgs, "org_type", t) for t in ["Company", "Education"]
    ]
    logging.info("Topic comparison")
    topic_comparison_chart, comp_table = make_chart_topic_comparison(
        topic_mix,
        arxiv_cat_lookup,
        [comp_ids, acad_ids],
        cats,
        ["company", "academia"],
        topic_list=topic_list,
        topic_category_map=topic_category_map,
        highlights=True,
        highlight_topics=labels_to_display,
    )

    save_highlights_table(comp_table, labels_to_display, "Company", topic_category_map)


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    driv = altair_visualisation_setup()
    main()
