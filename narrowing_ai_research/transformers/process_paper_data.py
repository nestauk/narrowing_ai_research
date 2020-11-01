import pandas as pd
import pickle
from narrowing_ai_research.utils.list_utils import flatten_list
import narrowing_ai_research
import datetime
import logging
import os
import json

project_dir = narrowing_ai_research.project_dir


def make_ai_ids():
    """Function to extract AI Ids from the categories and expanded paper
    list files
    """
    with open(f"{project_dir}/data/interim/find_ai_outputs.p", "rb") as infile:
        t = pickle.load(infile)

    paper_categories = pd.read_csv(
        f"{project_dir}/data/raw/arxiv_article_categories.csv"
    )

    ai_cats = set(["cs.AI", "cs.NE", "stat.ML", "cs.LG"])

    ai_core_papers = set(
        paper_categories.loc[paper_categories["category_id"].isin(ai_cats)][
            "article_id"
        ]
    )

    ai_papers_expanded = set(flatten_list([x for x in t[0].values()]))

    all_ai_ids = ai_core_papers.union(ai_papers_expanded)

    return all_ai_ids


def process_paper_data():
    """Some final data processing
    * Add AI dates to relevant datasets (papers and Grid)
    * Add dates to the papers df
    * Create long topic df
    * Add DeepMind and OpenAI papers to the paper_grid file
    """

    # Add dates
    # This reads the first line of the papers to check if year is there.
    # papers = pd.read_csv(
    #     f"{project_dir}/data/raw/arxiv_articles.csv", dtype={"article_id": str}
    # )
    if os.path.exists(f"{project_dir}/data/processed/arxiv_articles.csv") is True:
        logging.info("Already processed paper data")
        logging.info("Already added AI ids to data")
    else:
        papers = pd.read_csv(
            f"{project_dir}/data/raw/arxiv_articles.csv", dtype={"article_id": str}
        )

        ai_ids = make_ai_ids()

        logging.info("Adding dates to paper_df")
        papers["date"] = papers["created"].apply(
            lambda x: datetime.datetime(int(x.split("-")[0]), int(x.split("-")[1]), 1)
        )

        papers["year"] = papers["date"].apply(lambda x: x.year)

        logging.info("Add AI dummy")
        papers["is_ai"] = papers["article_id"].isin(ai_ids)

        papers.to_csv(f"{project_dir}/data/processed/arxiv_articles.csv", index=False)

        papers_year_dict = papers.set_index("article_id").to_dict()

    if os.path.exists(f"{project_dir}/data/processed/arxiv_topics_years.csv") is True:
        logging.info("Already created topic year df")

    else:
        logging.info("making topic year df")
        topic_mix = pd.read_csv(
            f"{project_dir}/data/processed/ai_topic_mix.csv", dtype={"article_id": str}
        )

        topic_long = topic_mix.melt(id_vars="article_id")

        topic_long["year"], topic_long["date"] = [
            [papers_year_dict[var][_id] for _id in topic_long["article_id"]]
            for var in ["created", "date"]
        ]

        topic_long.to_csv(
            f"{project_dir}/data/processed/arxiv_topics_years.csv", index=False
        )

    if os.path.exists(f"{project_dir}/data/processed/arxiv_grid.csv") is True:
        logging.info("Already processed GRID data")

    else:
        logging.info("Processing GRID data")
        logging.info("Fixing UCL bug")
        pd.options.mode.chained_assignment = None

        g = pd.read_csv(
            f"{project_dir}/data/raw/arxiv_grid_short.csv", dtype={"article_id": str}
        )

        ucl_aus = g.loc[g["institute_name"] == "UCL Australia"]

        ucl_aus["institute_name"] = "UCL"
        ucl_aus["institute_country"] = "United Kingdom"
        ucl_aus["institute_lat"] = 0.1340
        ucl_aus["institute_lon"] = 51.5246
        ucl_aus["org_type"] = "Education"

        g_no_aus = g.loc[g["institute_name"] != "UCL Australia"]

        g_fixed = pd.concat([g_no_aus, ucl_aus], axis=0)

        # g_fixed.to_csv("arxiv_grid_proc.csv",index=False)
        logging.info("Adding DeepMind and OpenAI")
        with open(f"{project_dir}/data/raw/scraped_arxiv.json", "r") as infile:
            scraped = json.load(infile)

        with open(f"{project_dir}/data/interim/scraped_meta.json", "r") as infile:
            scraped_meta = json.load(infile)

        scraped_c = {k.split("/")[-1]: v for k, v in scraped.items()}

        new_results = []

        # Create a df with the information for deepmind / openai ids
        scr_no_dupes = g.loc[
            [x in set(scraped_c.keys()) for x in g["article_id"]]
        ].drop_duplicates("article_id")

        # For each id there we create a new series with org metadata
        for _id, r in scr_no_dupes.iterrows():

            if r["article_id"] in scraped_c.keys():

                paper_vector = {}

                n = scraped_c[r["article_id"]]

                paper_vector["institute_name"] = n
                paper_vector["article_id"] = r["article_id"]
                paper_vector["mag_id"] = r["mag_id"]
                paper_vector["mag_authors"] = r["mag_authors"]
                paper_vector["is_multinational"] = 0
                paper_vector["institute_id"] = f"extra_{n}"
                paper_vector["institute_country"] = scraped_meta[n]["institute_country"]
                paper_vector["institute_lat"] = scraped_meta[n]["lat"]
                paper_vector["institute_lon"] = scraped_meta[n]["lon"]
                paper_vector["org_type"] = scraped_meta[n]["org_type"]

                paper_series = pd.Series(paper_vector)
                new_results.append(paper_series)

        grid_out = pd.concat([g_fixed, pd.DataFrame(new_results)], axis=0)

        logging.info("Adding AI labels")
        ai_ids = make_ai_ids()
        grid_out["is_ai"] = grid_out["article_id"].isin(ai_ids)

        logging.info("Saving grid file")
        grid_out.to_csv(f"{project_dir}/data/processed/arxiv_grid.csv", index=False)


if __name__ == "__main__":
    process_paper_data()

    os.remove(f"{project_dir}/data/raw/arxiv_articles.csv")

    os.remove(f"{project_dir}/data/raw/arxiv_grid_short.csv")
