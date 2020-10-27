import os
import pandas as pd
import joblib
import numpy as np
import narrowing_ai_research
from narrowing_ai_research.hSBM_Topicmodel.sbmtm import sbmtm
import logging
import yaml

project_dir = narrowing_ai_research.project_dir

# Parameters to filter topics
with open(f"{project_dir}/model_config.yaml", "r") as infile:
    filter_pars = yaml.safe_load(infile)["topic_filter"]
    pres = filter_pars["presence_threshold"]
    prev = filter_pars["prevalence_threshold"]

def post_process_model(model, top_level):
    """Function to post-process the outputs of a hierarchical topic model
    _____
    Args:
      model:      A hsbm topic model
      top_level:  The level of resolution at which we want to extract topics
    _____
    Returns:
      A topic mix df with topics and weights by document
    """
    # Extract the word mix (word components of each topic)
    logging.info("Creating topic names")
    word_mix = model.topics(l=top_level)

    # Create tidier names
    topic_name_lookup = {
        key: "_".join([x[0] for x in values[:5]]) 
        for key, values in word_mix.items()
    }
    topic_names = list(topic_name_lookup.values())

    # Extract the topic mix df
    logging.info("Extracting topics")
    topic_mix_ = pd.DataFrame(
        model.get_groups(l=top_level)["p_tw_d"].T,
        columns=topic_names,
        index=model.documents,
    )

    return topic_mix_

def filter_topics(topic_df, presence_thr, prevalence_thr):
    """Filter uninformative ("stop") topics
    Args:
        top_df (df): topics
        presence_thr (int): threshold to detect topic in article
        prevalence_thr (int): threshold to exclude topic from corpus
    Returns:
        Filtered df
    """
    # Remove highly uninformative / generic topics
    topic_prevalence = (
        topic_df.iloc[:, 1:]
        .applymap(lambda x: x > presence_thr)
        .mean()
        .sort_values(ascending=False)
    )
    # Filter topics
    filter_topics = topic_prevalence.index[
                    topic_prevalence > prevalence_thr].tolist()

    # We also remove short topics (with less than two ngrams)
    filter_topics = filter_topics + [
        x for x in topic_prevalence.index if len(x.split("_")) <= 2
    ]

    topic_df_filt = topic_df.drop(filter_topics, axis=1)

    return topic_df_filt, filter_topics

if __name__ == "__main__":

    if os.path.exists("{project_dir}/data/raw/ai_topic_mix.csv") is True:
        logging.info("Already created topic df")

    else:
        logging.info("loading data")
        m = joblib.load(f"{project_dir}/models/topsbm/ai_topsbm.pkl")
        out = post_process_model(m, 0)

        # Create to
        logging.info("Filtering topics")
        filt_df, filt_list = filter_topics(out, pres, prev)

        filt_df.to_csv(f"{project_dir}/data/raw/ai_topic_mix.csv")

        with open(f"{project_dir}/data/raw/filtered_topics.txt", "w") as outfile:
            outfile.write(", ".join(filt_list))
