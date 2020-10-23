A Narrowing of AI research?
==============================

Code for the Narrowing of AI research paper

## Setup

### Environment

Run `make create_environment` to setup the correct conda environment, and then `source activate narrowing_ai_research` or `conda_activate narrowing_ai_research` to activate it.

### Fetch data

Run `make data`.

This fetches data from the following sources:

* figshare:
  * rXiv data (papers, institutions and categories)
  * topics
  * abstract vectors
* GRID
* Scrape DeepMind and OpenAI arXiv IDs

Data is stored in `data/raw`

## Processing data

This:

* Cleans the fetched data
  * Fix UCL buggy match
  * Create dates etc
  * Adds DeepMind and OpenAI labels to relevant papers
* Trains a `word2vec` model on the abstracts
* Finds the AI papers in the data

All outputs are stored in `data/processed`

### Analysing the data

This reproduces the analysis and charts in the paper. 

All charts and tables are stored as pngs and htmls in the `reports/figures` folder.


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/nestauk/cookiecutter-data-science-nesta">Nesta cookiecutter data science project template</a>.</small></p>
