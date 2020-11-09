import logging
import yaml
import pandas as pd

from narrowing_ai_research.utils.read_utils import read_topic_mix, read_papers
from narrowing_ai_research.transformers.diversity import Diversity, remove_zero_axis


import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir


def topic_diversity_contribution(
    topic_mix, metric, params, name, method="max", threshold=None
):
    """Compares the diversity of the corpus without / without papers with a topic
    Args:
        topic mix: topic mix in papers
        metric: diversity metric to consider
        params: diversity parametres to consider
        name: name to label output
        method:
            max if we remove papers where topic is highest.
            pres is we remove papers where topic is above a threshold
    """

    # Calculate benchmark
    d = Diversity(topic_mix, name)

    getattr(d, metric)(params)

    bench = d.metric

    if method == "max":
        papers_max_topic = (
            topic_mix.idxmax(axis=1)
            .reset_index(name="topic")
            .groupby("topic")["article_id"]
            .apply(lambda x: set(x))
        )
    else:
        paper_bin = topic_mix.applymap(lambda x: x > threshold)

    results = []

    # For each topic we calculate the diversity without the topic
    # and subtract
    for n, t in enumerate(topic_mix.columns):
        if n % 20 == 0:
            logging.info(f"processed {n} topics")

        # With max method
        if method == "max":
            if t not in papers_max_topic.keys():
                presence = 0
                difference = 0
            else:
                presence = len(papers_max_topic[t])
                topic_mix_reduced = remove_zero_axis(
                    topic_mix.loc[~topic_mix.index.isin(papers_max_topic[t])]
                )

                d_ = Diversity(topic_mix_reduced, name)

                getattr(d_, metric)(params)
                reduced = d_.metric
                difference = bench - d_.metric
        # With presence method
        else:
            presence = paper_bin[t].sum()
            papers_without_topic = paper_bin.loc[paper_bin[t] == 0].index

            topic_mix_reduced = remove_zero_axis(topic_mix.loc[papers_without_topic])

            d_ = Diversity(topic_mix_reduced, name)

            getattr(d_, metric)(params)
            reduced = d_.metric
            difference = bench - reduced

        out = pd.Series(
            [t, presence, difference, metric, name],
            index=["topic", "presence", "div_contr", "metric", "parametre_set"],
        )
        results.append(out)

    return results


def topic_diversity_calculation_all(
    topic_mix, param_dict, method="max", threshold=None
):
    """Calculates topic contribution to diversity for all metrics and parametres
    Args:
        topic_mix: topic distribution
        param_dict: dict where keys are div metrics and values are parametres
        method: method to calculate contribution to diversity
        threshold: threshold for including a topic (only for pres)
    """

    all_results = []

    for k, v in param_dict.items():
        logging.info(k)

        for n, par in enumerate(v):
            results = topic_diversity_contribution(
                topic_mix, k, par, f"param_set_{n}", method=method, threshold=threshold
            )
            all_results.append(results)
    return all_results


def make_diversity_contribution_df(div_outputs, label):
    """Convers output of the diversity contribution function into a df"""

    df = pd.concat([pd.DataFrame(x) for x in div_outputs])
    df["diversity_contribution_method"] = label
    return df


def main():
    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        pars = yaml.safe_load(infile)

    div_params = pars["section_4"]["div_params"]
    section_pars = pars["section_6"]

    topic_mix = read_topic_mix()
    papers = read_papers()

    # Focus on recent AI papers
    papers_rec = papers.loc[
        (papers["year"] >= section_pars["year"]) & (papers["is_ai"] == True)
    ]

    topics_rec = remove_zero_axis(
        topic_mix.loc[
            topic_mix["article_id"].isin(set(papers_rec["article_id"]))
        ].set_index("article_id")
    )

    logging.info("Calculating diversity contributions")

    div_contr_max_df = make_diversity_contribution_df(
        topic_diversity_calculation_all(topics_rec, div_params), label="max"
    )
    div_contr_pres_df = make_diversity_contribution_df(
        topic_diversity_calculation_all(
            topics_rec, div_params, method="pres", threshold=section_pars["threshold"]
        ),
        label="presence",
    )

    diversity_contributions = pd.concat([div_contr_max_df, div_contr_pres_df])
    logging.info("Saving data")
    diversity_contributions.to_csv(
        f"{project_dir}/data/processed/diversity_contribution.csv", index=False
    )

if __name__=='__main__':
    main()
