import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import logging

from narrowing_ai_research.utils.read_utils import (
    read_papers,
    read_topic_mix,
    read_topic_category_map,
    read_arxiv_cat_lookup,
)
from narrowing_ai_research.transformers.networks import (
    make_co_network,
    plot_comp_network,
    network_distance_stats,
)

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir

matplotlib.rcParams["font.sans-serif"] = "Arial"


def read_process_data():
    papers = read_papers()
    topic_mix = read_topic_mix()
    topic_category_map = read_topic_category_map()
    arxiv_cat_lookup = read_arxiv_cat_lookup()

    return papers, topic_mix, topic_category_map, arxiv_cat_lookup


def main():

    with open(f"{project_dir}/paper_config.yaml", "r") as infile:
        params = yaml.safe_load(infile)["section_5"]

    papers, topic_mix, topic_category_map, arxiv_cat_lookup = read_process_data()

    logging.info("Network visualisation")
    net_1, size_1 = make_co_network(
        papers,
        topic_mix,
        topic_category_map,
        np.arange(params["network_1"]["y0"], params["network_1"]["y1"]),
        threshold=params["network_1"]["topic_threshold"],
    )

    net_2, size_2 = make_co_network(
        papers,
        topic_mix,
        topic_category_map,
        np.arange(params["network_2"]["y0"], params["network_2"]["y1"]),
        threshold=params["network_2"]["topic_threshold"],
    )

    my_cats = params["categories"]

    plot_comp_network(
        [net_1, net_2], [size_1, size_2], my_cats, arxiv_cat_lookup, topic_category_map
    )

    plt.tight_layout()
    plt.savefig(f"{project_dir}/reports/figures/png/fig_12_topic_network.png")

    logging.info("Network statistics")
    dists = pd.DataFrame(
        [
            network_distance_stats(net_1, "Network 2013-2016"),
            network_distance_stats(net_2, "Network 2019-2020"),
        ]
    )

    dists.to_latex(f"{project_dir}/reports/tables/network_metrics.tex")


if __name__ == "__main__":
    main()
