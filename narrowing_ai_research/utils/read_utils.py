import pandas as pd
import json
import logging
import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir


def read_papers():
    logging.info("Reading papers")
    papers = pd.read_csv(
        f"{project_dir}/data/processed/arxiv_articles.csv",
        dtype={"article_id": str},
        parse_dates=["date"],
        usecols=["article_id", "date", "is_ai"],
    )
    papers["year"] = [x.year for x in papers["date"]]
    return papers


def read_papers_orgs():
    logging.info("Reading papers-orgs")

    papers_orgs = pd.read_csv(
        f"{project_dir}/data/processed/arxiv_grid.csv", dtype={"article_id": str}
    )

    return papers_orgs


def read_topic_mix():
    logging.info("Reading topics")
    topic_mix = pd.read_csv(
        f"{project_dir}/data/processed/ai_topic_mix.csv", dtype={"article_id": str}
    )

    return topic_mix


def read_topic_long():
    logging.info("Reading topics years")
    topic_long = pd.read_csv(
        f"{project_dir}/data/processed/arxiv_topics_years.csv",
        dtype={"article_id": str},
        parse_dates=["date"],
        usecols=["article_id", "variable", "value", "date"],
    )

    return topic_long


def read_topic_category_map():
    logging.info("Reading topic - category map")
    with open(f"{project_dir}/data/interim/topic_category_map.json", "r") as infile:
        topic_category_map = json.load(infile)
    return topic_category_map


def read_arxiv_cat_lookup():
    logging.info("Reading arxiv category lookup")
    with open(f"{project_dir}/data/raw/arxiv_category_lookup.json", "r") as infile:
        arxiv_cat_lookup = json.load(infile)
    return arxiv_cat_lookup


def read_arxiv_categories():
    logging.info("Reading article categories")
    cats = pd.read_csv(
        f"{project_dir}/data/raw/arxiv_article_categories.csv",
        dtype={"article_id": str},
    )
    return cats


def query_orgs(paper_orgs, variable, name):
    """Returns ids for papers with a value in a variable"""

    _ids = paper_orgs.loc[paper_orgs[variable] == name]["article_id"]

    return set(_ids)


def paper_orgs_processing(paper_orgs, papers):
    """Additional processing of the paper orgs data"""
    p = paper_orgs.dropna(axis=0, subset=["institute_name"])

    logging.info("Clean institute names")

    p["org_name"] = [r[
            'institute_name'].split("(")[0].strip() if r['org_type']=='Company'
            else r['institute_name'] for rid,r in p.iterrows()]
    #p["institute_name"].apply(lambda x: x.split("(")[0].strip())

    logging.info("Drop duplicate institute - organisation pairs")
    # Enforce one paper - institute pair
    p_no_dupes = p.drop_duplicates(["article_id", "org_name"])

    keep_cols = [
        "article_id",
        "mag_authors",
        "org_type",
        "org_name",
        "institute_country",
        "is_ai",
    ]

    logging.info("Add dates")
    porgs = p_no_dupes[keep_cols].reset_index(drop=True)

    paper_date_dict = create_paper_dates_dict(papers)
    porgs["date"] = porgs["article_id"].map(paper_date_dict["date"])

    return porgs


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


def read_vectors():

    logging.info("Reading vectors")
    v = pd.read_csv(f"{project_dir}/data/raw/ai_vectors.csv")
    return v
