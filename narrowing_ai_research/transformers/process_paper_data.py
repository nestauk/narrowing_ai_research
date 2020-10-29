import pandas as pd
import narrowing_ai_research
import datetime
import logging
import os
import json

project_dir = narrowing_ai_research.project_dir


def process_paper_data():
    """Some final data processing
    * Add dates to the papers df
    * Create long topic df
    * Add DeepMind and OpenAI papers to the paper_grid file


    """

    # Add dates
    # This reads the first line of the papers to check if year is there.
    papers = pd.read_csv(
        f"{project_dir}/data/raw/arxiv_articles.csv", dtype={"article_id": str}
    )

    if "year" in papers.columns:
        logging.info("Already created paper dates")

    else:
        logging.info("Adding dates to paper_df")

        papers["date"] = papers["created"].apply(
            lambda x: datetime.datetime(int(x.split("-")[0]), int(x.split("-")[1]), 1)
        )

        papers["year"] = papers["date"].apply(lambda x: x.year)

        papers.to_csv(f"{project_dir}/data/raw/arxiv_articles_d.csv", index=False)

        papers_year_dict = papers.set_index("id").to_dict()

    if os.path.exists(f"{project_dir}/data/raw/arxiv_topics_years.csv") is True:
        logging.info("topic year df already created")

    else:
        logging.info("Reading data")

        topic_mix = pd.read_csv(
            f"{project_dir}/data/raw/ai_topic_mix.csv", dtype={"article_id": str}
        )

        topic_long = topic_mix.melt(id_vars="article_id")

        topic_long["year"], topic_long["date"] = [
            [papers_year_dict[var][_id] for _id in topic_long["article_id"]]
            for var in ["created", "date"]
        ]

    topic_long.to_csv(f"{project_dir}/data/raw/arxiv_topics_years.csv", index=False)

    if os.path.exists(f"{project_dir}/data/raw/arxiv_grid_proc.csv") is True:
        logging.info("Grid data already processed")

    else:
        logging.info("Processing GRID data: fixing UCL bad match")
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
        logging.info("Processing GRID data: adding DeepMind and OpenAI")
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

        logging.info("Saving grid file")
        grid_out.to_csv(f"{project_dir}/data/raw/arxiv_grid_proc.csv", index=False)

        os.remove(f"{project_dir}/data/raw/arxiv_grid_short.csv")
