import pandas as pd
import numpy as np
import altair as alt
import logging
import os
import yaml
from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import zscore

from narrowing_ai_research.utils.altair_utils import (
    altair_visualisation_setup,
    save_altair,
)
from narrowing_ai_research.paper.s3_org_eda import paper_orgs_processing
from narrowing_ai_research.paper.make_org_diversity import make_org_diversity

from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_papers_orgs,
)

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir

pd.options.mode.chained_assignment = None
alt.data_transformers.disable_max_rows()


def read_process_data():
    papers = read_papers()
    papers_orgs = paper_orgs_processing(read_papers_orgs(), papers)
    # Date
    papers_orgs["year"] = [x.year for x in papers_orgs["date"]]

    # Org diversity df

    org_div_df = f"{project_dir}/data/processed/org_diversity.csv"
    if os.path.exists(org_div_df) is False:
        logging.info("Making organisational diversity")
        make_org_diversity()
        org_diversity = pd.read_csv(f"{project_dir}/data/processed/org_diversity.csv")

    else:
        logging.info("Reading organisational diversity")
        org_diversity = pd.read_csv(f"{project_dir}/data/processed/org_diversity.csv")

    return papers, papers_orgs, org_diversity


def make_regression_dataset(papers_orgs, org_diversity, google_filter=True):

    # Lookup between org name and org type
    org_name_type_lookup = (
        papers_orgs.drop_duplicates("org_name")
        .set_index("org_name")["org_type"]
        .to_dict()
    )

    # Relevant papers
    papers_rel = papers_orgs.loc[
        (papers_orgs["is_ai"] == True) & (papers_orgs["year"] >= 2018)
    ]

    # Paper counts in the period
    paper_counts_all = papers_rel["org_name"].value_counts()
    paper_counts_all = paper_counts_all.append(
        pd.Series(
            paper_counts_all.Google - paper_counts_all.DeepMind,
            index=["Google_wo_DeepMind"],
        )
    )

    # Lookup between org name and paper year
    paper_counts = (
        papers_rel.groupby(["year", "org_name"])["article_id"].count().to_dict()
    )

    # If we are filtering Google, we remove DeepMind papers from them
    if google_filter is True:
        for y in range(2018, 2021):
            paper_counts[(y, "Google_wo_DeepMind")] = (
                paper_counts[(y, "Google")] - paper_counts[(y, "DeepMind")]
            )

        # Add the Google wo DeepMind information
        org_name_type_lookup["Google_wo_DeepMind"] = "Company"

    org_diversity["organisation_type"] = org_diversity["category"].map(
        org_name_type_lookup
    )
    org_diversity["number_of_papers"] = [
        paper_counts[(r["year"], r["category"])] for rid, r in org_diversity.iterrows()
    ]

    if google_filter is True:
        org_diversity = org_diversity.loc[org_diversity["category"] != "Google"]
    else:
        org_diversity = org_diversity.loc[
            org_diversity["category"] != "Google_wo_DeepMind"
        ]

    return org_diversity, org_name_type_lookup, paper_counts_all


def fit_regression(org_diversity, cov_type="HC1"):
    """
    Fits regression model
    """
    # Metrics

    # Storage for results
    reg_results = {"balance": {}, "weitzman": {}, "rao_stirling": {}}

    # For each variable we subset by metric and parametre set
    for v in ["balance", "weitzman", "rao_stirling"]:

        met = org_diversity.loc[org_diversity["metric"] == v]

        for m in [0, 1, 2]:
            reg_results[v][m] = {}

            # Subset
            div = met.loc[met["parametre_set"] == f"Parametre Set {m}"]

            # Normalise variables
            Y = zscore(div["score"]).astype(float)

            # Create company dummy
            div["is_company"] = div["organisation_type"] == "Company"

            # Papers logged
            div["papers_log"] = np.log(div["number_of_papers"])

            # Create org fixed effects
            fe = pd.get_dummies(div["category"])

            # Endogenous without fixed effects
            X_no_fe = add_constant(div[["is_company", "papers_log", "year"]]).astype(
                float
            )

            # Endogenous with fixed effects
            X_fe = add_constant(
                pd.concat(
                    [div[["is_company", "papers_log", "year"]], fe], axis=1
                ).astype(float)
            )

            # For both endogenous fit the models
            for X, n in zip([X_no_fe, X_fe], ["no_fe", "fe"]):
                ols = OLS(Y, X).fit(cov_type=cov_type)
                reg_results[v][m][n] = ols

    return reg_results


def make_regression_table(
    reg_results,
    org_names_list,
    save=True,
    metrics=["balance", "weitzman", "rao_stirling"],
):
    """Creates a regression table for the paper
    Also returns organisational coefficients used in a visualisation later
    """
    tables = []

    org_coeffs = {}

    for k, v in reg_results.items():
        for k2, v2 in v.items():
            for k3, v3 in v2.items():

                var_names = ["is_company", "papers_log", "year"]

                params = []
                names = []
                for v in var_names:

                    par = np.float(v3.params[v])
                    t = np.float(v3.tvalues[v])
                    pv = np.float(v3.pvalues[v])

                    if pv < 0.01:
                        par = str(np.round(par, 2)) + "***"
                    elif pv < 0.05:
                        par = str(np.round(par, 2)) + "**"
                    elif pv < 0.1:
                        par = str(np.round(par, 2)) + "*"
                    else:
                        par = str(np.round(par, 2)) + "*"

                    params.append(par)
                    params.append(f"({str(round(t,2))})")

                    names.append(v)
                    names.append(v + "_t_val")

                main_results = pd.Series(params, index=names)

                fe = "Yes" if k3 == "fe" else "No"
                other_details = pd.Series(
                    [np.round(v3.rsquared, 2), int(v3.nobs), fe],
                    index=["$R^2$", "obs", "FE"],
                )

                results_series = pd.concat([main_results, other_details])

                results_series.name = "_".join([k, str(k2)])

                if k3 == "fe":

                    org_vals = pd.concat(
                        [
                            v3.params.loc[org_names_list],
                            v3.conf_int()[0].loc[org_names_list],
                            v3.conf_int()[1].loc[org_names_list],
                        ],
                        axis=1,
                    )

                    org_vals.columns = ["beta", "lower", "upper"]
                    org_coeffs["_".join([k, str(k2)])] = org_vals

                    tables.append(results_series)

    reg_results_table = pd.concat(tables, axis=1)

    if save is True:
        clean_name_lookup = {
            "is_company": "Company index",
            "papers_log": "Papers (log)",
            "year": "Year",
            "$R^2$": "$R^2$",
            "obs": "N",
            "FE": "Fixed Effects",
        }

        for m in metrics:
            rel_table = reg_results_table.loc[
                :, [m in x for x in reg_results_table.columns]
            ]

            rel_table.columns = [x.split("_")[-1] for x in rel_table.columns]
            rel_table = rel_table.reset_index(drop=False)

            rel_table["index"] = [
                clean_name_lookup[x] if x in clean_name_lookup.keys() else ""
                for x in rel_table["index"]
            ]

            rel_table.rename(columns={"index": m}, inplace=True)

            rel_table.to_latex(f"{project_dir}/reports/tables/{m}.tex", index=False)

    return reg_results_table, org_coeffs


def make_chart_organisational_diversity(
    org_coeffs,
    num_orgs,
    metric_params,
    org_type_lookup,
    paper_counts,
    save=True,
    fig_num=14,
):
    """Plot comparing the organisational diversity coefficients"""

    # Regression coefficients sorted
    selected = (
        org_coeffs[metric_params]
        .sort_values("beta")
        .head(n=num_orgs)
        .reset_index(drop=False)
    )

    selected["org_type"] = selected["index"].map(org_type_lookup)
    selected["order"] = range(0, len(selected))

    # Paper counts by organisation
    recent_papers_orgs = (
        paper_counts.loc[selected["index"]]
        .reset_index(name="papers")
        .rename(columns={"index": "org"})
    )
    recent_papers_orgs["order"] = range(0, len(recent_papers_orgs))
    recent_papers_orgs["org_type"] = recent_papers_orgs["org"].map(org_type_lookup)

    b_ch = (
        alt.Chart(selected)
        .mark_bar()
        .encode(
            y=alt.Y("index", sort=alt.EncodingSortField("order"), title=""),
            x=alt.X("beta", title="Coefficient on diversity"),
            color=alt.X("org_type", title="Organisation type"),
        )
    ).properties(width=150, height=600)

    b_err = (
        alt.Chart(selected)
        .mark_errorbar()
        .encode(
            y=alt.Y(
                "index",
                sort=alt.EncodingSortField("order"),
                title="",
                axis=alt.Axis(ticks=False, labels=False),
            ),
            x=alt.X("lower", title=""),
            x2="upper",
        )
    ).properties(width=150, height=600)

    b_act = (
        alt.Chart(recent_papers_orgs)
        .mark_bar()
        .encode(
            y=alt.Y(
                "org",
                title=None,
                sort=alt.EncodingSortField("order"),
                axis=alt.Axis(labels=False, ticks=False),
            ),
            x=alt.X("papers"),
            color="org_type",
        )
    ).properties(width=100, height=600)

    out = (b_ch + b_err).resolve_scale(y="independent")
    out_2 = alt.hconcat(out, b_act, spacing=0).resolve_scale(y="shared")

    if save is True:
        save_altair(out_2, f"fig_{fig_num}_comp", driv)

    return out_2


def main():
    # We use the same diversity parametres as in the analysis of diversity
    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        pars = yaml.safe_load(infile)["section_8"]

    papers, papers_orgs, org_diversity = read_process_data()

    logging.info("Creating dataset for regression")
    reg_df, org_name_type_lookup, paper_counts_all = make_regression_dataset(
        papers_orgs, org_diversity, google_filter=pars["google_filter"]
    )
    logging.info("Fitting regression")
    reg_results = fit_regression(reg_df)

    logging.info("Saving regression results")
    org_names_list = set(reg_df["category"])
    tab, org_coeffs = make_regression_table(reg_results, org_names_list, save=True)

    logging.info("Organisational chart")
    div_chart = make_chart_organisational_diversity(
        org_coeffs,
        60,
        "rao_stirling_2",
        org_name_type_lookup,
        paper_counts_all,
        save=True,
    )


if __name__ == "__main__":
    driv = altair_visualisation_setup()
    main()
