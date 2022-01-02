# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# # Fetch AI paper citations

# +
# Preamble

# %load_ext autoreload
# %autoreload 2
# %config Completer.use_jedi = False
# -

import logging
from itertools import chain
from numpy.random import choice
from narrowing_ai_research import project_dir
from narrowing_ai_research.utils.read_utils import read_arxiv_categories
import json
import pickle
import requests
import numpy as np
import pandas as pd
import ratelim

# +
core_ai_cats = ["cs.AI","cs.NE","cs.LG","stat.ML"]

def get_ai_ids():
    
    with open(f"{project_dir}/data/interim/find_ai_outputs.p",'rb') as infile:
        expanded_ids = set(chain(*[vals for vals in pickle.load(infile)[0].values()]))
        
    paper_cats = read_arxiv_categories()
    
    core_ids = set(paper_cats.loc[paper_cats['category_id'].isin(core_ai_cats)]['article_id'])
    
    return expanded_ids | core_ids


@ratelim.patient(1,3.75)
def fetch_semantic_scholar(arxiv_id):
    
    base = "https://api.semanticscholar.org/v1/paper/arXiv:{}"
    
    response = requests.get(base.format(arxiv_id))
    
    if response.status_code == 200:
        return response.json()
    
    else:
        logging.info(f"{arxiv_id} status {response.status_code}")
    


# +
# Read AI paper ids

ai_ids = get_ai_ids()

# +
# Fetch citations from a random sample

# Random sample
ai_sample = choice(list(ai_ids),10000,replace=False)
# -

# Get citations
ss_results = []

for n,p in enumerate(ai_sample[9536:]):
    if n % 100 == 0:
        logging.info(n)
    
    result = fetch_semantic_scholar(p)
    
    ss_results.append(result)

# Save them
with open(f"{project_dir}/data/raw/ai_semantic_results.json",'w') as outfile:
    json.dump(ss_results,outfile)

len(ss_results)

# +
# Count citations out of the initial set

# +
# Compare topic distribution of in-corpus v out-corpus
# -

300/80

0%10


