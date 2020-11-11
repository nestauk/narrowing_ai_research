A Narrowing of AI research?
==============================

Code for the Narrowing of AI research paper ([arXiv link](https://arxiv.org/abs/2009.10385)) where we analyse the evolution of thematic diversity in the corpus of AI research in arXiv.

## Setup

### Environment

Run `make create_environment` to setup the correct conda environment, and then `source activate narrowing_ai_research` or `conda_activate narrowing_ai_research` to activate it.

Run `conda install -c conda-forge graph-tool` to install `graph-tool`.

### Fetch data

Run `make fetch`.

This fetches data from the following sources:

* [Figshare](https://figshare.com/account/home#/projects/91427):
  * rXiv data (papers, institutions and categories)
  * topics
  * abstract vectors
* Scrapes arXiv ids from DeepMind and OpenAI websites

Data is stored in `data/raw`

It also fetches a topic model trained on arXiv abstracts identified as AI. The model is stored in `models/topsbm`

## Processing data

Run `make data`

This:

* Cleans the fetched data
  * Create dates etc
  * Adds DeepMind and OpenAI labels to relevant papers
* Trains a `word2vec` model on the abstracts
* Finds the AI papers in the data and labels relevant datasets with an AI dummy
* Extracts topics form the topic model trained offline and removes generic and uninformative topics

All data outputs are stored in `data/processed`
To save space, processed files in `data/raw` are removed.

### Analysing the data

Run the scripts in `narrowing_ai_research/paper` to reproduce sections in the paper.

This:
* Reproduces all results in the paper including
  * Figures (in html and png) (in `reports/figures`)
  * Tables (in LaTeX) (in `reports/tables`)
  * Results (a text file in `reports`)
* You can also explore the results in the `notebooks/paper` folder.

The notebooks import functions from scripts in the `narrowing_ai_research/paper` folder.

The above uses the parametres in `paper_config.yaml`. Change for robustness tests etc.

Note that there are two steps in the pipeline that are not included in the repo. They are:

1. Fitting of hierarchical topic model (requires a machine w/ 64Gb RAM)
2. Creation of article embeddings

Step 1 is relatively easy to reproduce using the `arxiv_tokenised.json` tokenised abstracts generated while processing the data and the `smbtm` package that is already installed (for more infomation see [here](https://topsbm.github.io/)).



--------

<p><small>Project based on the <a target="_blank" href="https://github.com/nestauk/cookiecutter-data-science-nesta">Nesta cookiecutter data science project template</a>.</small></p>
