A Narrowing of AI research?
==============================

Code for the Narrowing of AI research paper ((arXiv link)[https://arxiv.org/abs/2009.10385])

## Setup

### Environment

Run `make create_environment` to setup the correct conda environment, and then `source activate narrowing_ai_research` or `conda_activate narrowing_ai_research` to activate it.

Run `conda install -c conda-forge graph-tool` to install `graph-tool`.

### Fetch data

Run `make fetch`.

This fetches data from the following sources:

* figshare:
  * rXiv data (papers, institutions and categories)
  * topics
  * abstract vectors
* Scrapes arXiv ids from DeepMind and OpenAI websites

Data is stored in `data/raw`

It also fetches a topic model trained on arXiv abstracts identified as AI

The model is stored in `models/topsbm`

## Processing data

This:

* Cleans the fetched data
  * Create dates etc
  * Adds DeepMind and OpenAI labels to relevant papers
* Trains a `word2vec` model on the abstracts
* Finds the AI papers in the data

All data outputs are stored in `data/processed`

### Analysing the data

This reproduces the analysis and charts in the paper. 

All charts and tables are stored as pngs and htmls in the `reports/figures` folder.


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/nestauk/cookiecutter-data-science-nesta">Nesta cookiecutter data science project template</a>.</small></p>
