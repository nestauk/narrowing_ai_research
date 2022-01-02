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

# # Robustness of AI identification mechanism

# +
# %load_ext autoreload
# %autoreload 2

# %config Completer.use_jedi = False
# -

import altair as alt
import json
import numpy as np
import pandas as pd
from narrowing_ai_research import project_dir
from narrowing_ai_research.utils.read_utils import get_ai_ids, read_tokenised, read_papers
from typing import List
from itertools import chain
from toolz import pipe
import tomotopy as tp
from numpy.random import choice

# ### Overlap between AI corpus and highly citing / referencing papers

# +
with open(f"{project_dir}/data/raw/ai_semantic_results.json",'r') as infile:
    ss = json.load(infile)

ai_ids = get_ai_ids()
    
# -

def get_citation_ids(article:dict,direction:str='citations')->List:
    """Get ids for papers cited / citing a paper in our corpus"""
    
    return [c['arxivId'] for c in article[direction]]
    


# +
# Remove a few "none" semantic scholar results
NoneType = type(None)
ss = [art for art in ss if type(art) != NoneType]

# Get citation ids and references ids
citations_ids, references_ids = [list(chain(*[get_citation_ids(art,direct) for art in ss])) for direct in ['citations','references']]


# +
# How many are NoneType / arXiv
def share_of_nones(citation_result:list)->float:
    return citation_result.count(None)/len(citation_result)

def corpus_overlap(cit_ids:list, ai_ids:set=ai_ids, ranks:list = [0,10,100,1000,10000])->pd.DataFrame:
    '''Returns the overlap between a list of citations and the AI corpus.
    This is a dataframe with overlap by frequency quintile and in total 
    '''
    # Citation rates by position in the distribution
    return (pd.Series([_id for _id in cit_ids if _id != None])
                .value_counts()
                .to_frame(name='freq')
                .reset_index(drop=False)
                .rename(columns={'index':'arxiv_id'})
                .assign(position = lambda df: pd.cut(df.index, bins=ranks+[max(df.index)],include_lowest=True,labels=False))
                .assign(in_ai_ids = lambda df: [_id in ai_ids for _id in df['arxiv_id']])
                .groupby('position')['in_ai_ids'].mean())
    
    
def get_citation_meta(article:dict, direction:str='citations')->pd.Series:
    
    return article[direction]


# +
### We want to remove from citations papers published after 2020: by definition those papers couldn't be in our corpus

def get_citation_meta(article:dict, direction:str='citations')->dict:
    
    return article[direction]

all_citations = pd.DataFrame(chain(*[get_citation_meta(art, direction='citations') for art in ss]))[['arxivId','year']]

pre_2021_papers = pipe(pd.DataFrame(chain(*[get_citation_meta(art) for art in ss]))
                   [['arxivId','year']]
                   .query("year<=2020")
                   ['arxivId'].tolist(),set)



# +
overlap_results = pd.concat([corpus_overlap(cit_ids).to_frame().assign(direction=name) for cit_ids,name in 
                            zip([pre_2021_papers,references_ids],['citations','references'])],axis=0).reset_index(drop=False)

overlap_chart = (alt.Chart(overlap_results)
                 .mark_line(point=True,strokeWidth=1)
                 .encode(x=alt.X('position:N',title='Position in citation distribution'),
                         y=alt.Y('in_ai_ids',axis=alt.Axis(format='%'),title='overlap with AI corpus'),
                         color=alt.Color('direction',title='Relation to sample')
                )).properties(height=200,width=400)

overlap_chart


# +
# Interesting: most of the papers refer to each other but are cited by papers outside. Now we need to check the semantic differences
# between citing papers outside of our corpus and referenced papers inside our corpus. We will do this using tomotopy.
# -

# ### Topic modelling comparison

# +
def get_topic_words(topic, top_words=5):
    """Extracts main words for a topic"""

    return "_".join([x[0] for x in topic[:top_words]])


def make_topic_mix(mdl, num_topics, doc_indices):
    """Takes a tomotopy model and products a topic mix"""
    topic_mix = pd.DataFrame(
        np.array([mdl.docs[n].get_topic_dist() for n in range(len(doc_indices))])
    )

    topic_mix.columns = [
        get_topic_words(mdl.get_topic_words(n, top_n=5)) for n in range(num_topics)
    ]

    topic_mix.index = doc_indices
    return topic_mix


# -

# Tasks:
# Read tokenised abstracts
tok = read_tokenised()

# +
# Create training corpus (we choose 30,000 AI papers to speed things up and preserve balance in the training set)

# +
sample_ai_ids = set(choice(list(ai_ids),30000, replace=False))
reference_exc = set(references_ids)-sample_ai_ids
citation_exc = set(pre_2021_papers)-sample_ai_ids

selected_ids = sample_ai_ids | reference_exc | citation_exc 

# Removing empty documents
selected_corpus = {k:v for k,v in tok.items() if k in selected_ids and len(v)>0}

# +
# Train a topic model and extract the topic mix

mdl = tp.LDAModel(k=300)
for key,doc in list(selected_corpus.items()):
    mdl.add_doc(doc)

for i in range(0, 150, 10):
    mdl.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

# for k in range(mdl.k):
#     print('Top 10 words of topic #{}'.format(k))
#     print(mdl.get_topic_words(k, top_n=10))

mdl.summary()


# Calculate topic distributions by category and compare with AI: are they significantly different?
# We could check this by comparing the means of the log of 
# -

index_category_lookup = (pd.concat(
    [pd.DataFrame({'arxiv_id':list(ref),'category':len(ref)*[name]}) for ref,name in zip([sample_ai_ids,reference_exc,
                                                                                          citation_exc],
                                                                                         ['ai','reference','citations'])]
            ).reset_index(drop=True))

topic_mix = (make_topic_mix(mdl,300,list(selected_corpus.keys()))
             .stack()
             .reset_index(drop=False)
             .rename(columns={'level_0':'arxiv_id','level_1':'topic',0:'weight'})
             .merge(index_category_lookup,on='arxiv_id'))             

topic_cat_means = topic_mix.groupby(['category','topic'])['weight'].mean()

topic_cat_means.unstack(level=0).corr()

topic_plot = (alt.Chart(topic_cat_means_plot.reset_index(drop=False))
              .mark_point(filled=True, shape='square')
              .encode(x=alt.X('topic',axis=alt.Axis(labels=False,ticks=False),title='Topic',
                              sort=alt.EncodingSortField('weight',order='descending')),
                      y=alt.Y('weight',scale=alt.Scale(type='log'),title='Mean topic weight'),
                      color=alt.Color('category',title='Article category'),
                      tooltip=['topic','category','weight'])
             ).properties(width=600,height=300)

topic_plot

# ### Top differences between groups

# +
#omparisons = [['ai','reference'], ['ai','citations']]
# -

topic_mix_pop = topic_mix.groupby('topic')['weight'].mean()

# +
topic_mean_comp = (topic_cat_means
                   .unstack(level=0)
                   .apply(lambda x: x/topic_mix_pop))

table = {'category':[],'top_topics':[]}

for cat in topic_mean_comp.columns:
    
    table['category'].append(cat)
    
    table['top_topics'].append(", ".join(topic_mean_comp[cat].sort_values(ascending=False)[:15].index))
# -

pd.DataFrame(table)
