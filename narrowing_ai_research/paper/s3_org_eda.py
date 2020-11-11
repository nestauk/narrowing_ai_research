import pandas as pd
import logging
import altair as alt

import narrowing_ai_research
from narrowing_ai_research.utils.altair_utils import (
    altair_visualisation_setup,
    save_altair,
)

from narrowing_ai_research.utils.read_utils import read_papers, read_papers_orgs

project_dir = narrowing_ai_research.project_dir


def read_process_data():

    papers = read_papers()
    paper_orgs = read_papers_orgs()

    return papers, paper_orgs


def create_paper_dates_dict(papers):
    """Creates a dict between paper and year and date of creation
    Args:
        papers: dataframe with paper metadata
    """
    p = pd.DataFrame()
    p["article_id"] = papers["article_id"]

    p["year"] = [x.year for x in papers["date"]]
    p["date"] = [x for x in papers["date"]]

    papers_year_dict = p.set_index("article_id")[["date", "year"]].to_dict()

    return papers_year_dict


def paper_orgs_processing(paper_orgs, papers):
    """Additional processing of the paper orgs data"""
    p = paper_orgs.dropna(axis=0, subset=["institute_name"])

    logging.info("Clean institute names")

    p["org_name"] = p["institute_name"].apply(lambda x: x.split("(")[0].strip())

    logging.info("Drop duplicate institute - organisation pairs")
    # Enforce one paper - institute pair
    p_no_dupes = p.drop_duplicates(["article_id", "org_name"])

    keep_cols = ["article_id", "mag_authors", "org_type", "org_name", "is_ai"]

    logging.info("Add dates")
    porgs = p_no_dupes[keep_cols].reset_index(drop=True)

    paper_date_dict = create_paper_dates_dict(papers)
    porgs["date"] = porgs["article_id"].map(paper_date_dict["date"])

    return porgs


def make_chart_type_comparison(porgs, save=True, fig_number=6):
    """Plots evolution of activity by organisation type"""
    # Counts activity by type
    org_type_count = (
        porgs.groupby(["is_ai", "org_type"]).size().reset_index(name="count")
    )

    # Melts and normalises
    org_type_long = (
        org_type_count.pivot_table(index="org_type", columns="is_ai", values="count")
        .apply(lambda x: 100 * x / x.sum())
        .reset_index(drop=False)
        .melt(id_vars="org_type")
    )

    # Add clean AI name
    org_type_long["category"] = [
        "AI" if x is True else "Not AI" for x in org_type_long["is_ai"]
    ]

    # Create altair base
    base = alt.Chart(org_type_long).encode(
        y=alt.Y(
            "org_type",
            title=["Type of", "organisation"],
            sort=alt.EncodingSortField("value", "sum", order="descending"),
        ),
        x=alt.X("value", title="% activity"),
    )

    type_comp_point = base.mark_point(filled=True).encode(
        color="category", shape="category"
    )

    type_comp_l = base.mark_line(
        stroke="grey", strokeWidth=1, strokeDash=[1, 1]
    ).encode(detail="org_type")

    type_comp_chart = (type_comp_point + type_comp_l).properties(height=100)

    if save is True:
        save_altair(type_comp_chart, f"fig_{fig_number}_type_comp", driv)

    return type_comp_chart


def make_chart_type_evol(porgs, save=True, fig_number=7):
    """Plots evolution of org types"""

    # Evolution of types
    org_type_evol = (
        porgs.groupby(["is_ai", "org_type", "date"]).size().reset_index(name="count")
    )

    org_type_evol_wide = org_type_evol.pivot_table(
        index=["date", "is_ai"], columns="org_type", values="count"
    ).fillna(0)

    # Calculate shares
    org_type_evol_sh = org_type_evol_wide.apply(lambda x: x / x.sum(), axis=1)

    # Melt and clean
    org_type_evol_long = org_type_evol_sh.reset_index(drop=False).melt(
        id_vars=["is_ai", "date"]
    )

    org_type_evol_long = org_type_evol_long.loc[
        [x.year > 2000 for x in org_type_evol_long["date"]]
    ]

    org_type_evol_long["category"] = [
        "AI" if x is True else "Not AI" for x in org_type_evol_long["is_ai"]
    ]

    # Visualise
    org_type_evol_ch = (
        (
            alt.Chart(org_type_evol_long)
            .transform_window(
                m="mean(value)", frame=[-10, 0], groupby=["is_ai", "org_type"]
            )
            .transform_filter(
                alt.FieldOneOfPredicate(
                    "org_type", ["Company", "Nonprofit", "Government", "Healthcare"]
                )
            )
            .mark_line()
            .encode(
                x=alt.X("date:T", title=""),
                y=alt.Y("m:Q", title="Share of activity"),
                color=alt.Color(
                    "org_type",
                    title="Type of organisation",
                    sort=alt.EncodingSortField("count", "sum", order="descending"),
                ),
                column=alt.Column(
                    "category", title="Research category", sort=["Not AI", "AI"]
                ),
            )
        )
        .resolve_scale(y="independent")
        .properties(width=200, height=150)
    )

    if save is True:
        save_altair(org_type_evol_ch, f"fig_{fig_number}_type_evol", driv)

    return org_type_evol_ch


def make_chart_company_activity(
    porgs, papers, top_c=15, t0=2005, roll_w=-8, save=True, fig_num=8
):
    """Chart evolution of company activity"""
    logging.info("Preparing data")
    # Create a paper - date lookup
    paper_year_date = create_paper_dates_dict(papers)["date"]

    # Find paper IDs for top 15 companies
    top_comps = (
        porgs.loc[(porgs["is_ai"] == True) & (porgs["org_type"] == "Company")][
            "org_name"
        ]
        .value_counts()
        .head(n=top_c)
        .index
    )

    comp_papers = {
        org: set(porgs.loc[porgs["org_name"] == org]["article_id"]) for org in top_comps
    }

    # Concatenate them in a dataframe
    comp_trends = (
        pd.DataFrame(
            [
                pd.Series(
                    [paper_year_date[x] for x in ser if x in paper_year_date.keys()],
                    name=n,
                ).value_counts()
                for n, ser in comp_papers.items()
            ]
        )
        .fillna(0)
        .T
    )

    comp_trends_long = (
        comp_trends.reset_index(drop=False)
        .melt(id_vars="index")
        .assign(indicator="Total AI papers")
    )

    # Extract top 20 comps for 2020 (we use this to order the chart later)
    comps_2020 = (
        comp_trends.loc[[x.year == 2020 for x in comp_trends.index]]
        .sum()
        .sort_values(ascending=False)
    )
    comps_order = {n: name for n, name in enumerate(comps_2020.index)}

    # Normalise with paper counts for all AI
    logging.info("Normalising")
    all_ai_counts = (
        porgs.drop_duplicates("article_id")
        .query("is_ai == True")["date"]
        .value_counts()
    )

    comp_trends_norm = comp_trends.apply(lambda x: x / all_ai_counts).fillna(0)

    comp_trends_share_long = (
        comp_trends_norm.reset_index(drop=False)
        .melt(id_vars="index")
        .assign(indicator="Share of all AI")
    )

    # Some tidying up before plotting
    comp_trends = pd.concat([comp_trends_long, comp_trends_share_long])
    comp_trends_recent = comp_trends.loc[[(x.year > t0) for x in comp_trends["index"]]]
    comp_trends_recent["order"] = comp_trends_recent["variable"].map(comps_order)

    date_domain = list(pd.to_datetime(["2007-01-01", "2020-07-01"]))

    logging.info("Plotting")
    # Create chart
    comp_evol_chart = (
        (
            alt.Chart(comp_trends_recent)
            .mark_area(stroke="black", strokeWidth=0.1, clip=True)
            .transform_window(
                roll="mean(value)", frame=[roll_w, 0], groupby=["indicator"]
            )
            .encode(
                x=alt.X("index:T", title="", scale=alt.Scale(domain=date_domain)),
                y=alt.Y("roll:Q", title=""),
                color=alt.Color(
                    "variable",
                    title="Company",
                    scale=alt.Scale(scheme="tableau20"),
                    sort=comps_2020.index.tolist(),
                ),
                order=alt.Order("order"),
                row=alt.Row("indicator", sort=["total", "share"]),
            )
        )
        .properties(height=200)
        .resolve_scale(y="independent")
    )

    if save is True:
        save_altair(comp_evol_chart, f"fig_{fig_num}_company_evol", driv)

    return comp_evol_chart


def main():
    papers, paper_orgs = read_process_data()

    porgs = paper_orgs_processing(paper_orgs, papers)

    # Analysis
    logging.info("Comparing org type activities")
    comp = make_chart_type_comparison(porgs, save=True)

    logging.info("Comparing org type trends")
    type_evol = make_chart_type_evol(porgs, save=True)

    logging.info("Comparing company activities")
    comp_trends = make_chart_company_activity(porgs, papers, save=False)


if __name__ == "__main__":
    driv = altair_visualisation_setup()
    main()
