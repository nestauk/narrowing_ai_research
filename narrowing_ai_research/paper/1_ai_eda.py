import numpy as np
import scipy as sp
import pandas as pd
import logging
import json
import pickle
import random
import altair as alt
from altair_saver import save
from itertools import chain


import narrowing_ai_research
from narrowing_ai_research.utils.list_utils import flatten_list, flatten_freq
from narrowing_ai_research.utils.altair_utils import altair_visualisation_setup, save_altair


project_dir = narrowing_ai_research.project_dir

def load_process_data():
    ''' Loads AI paper data for analysis in section 1.
    '''
    logging.info("Loading data")
    logging.info("Loading papers")
    
    papers = pd.read_csv(f"{project_dir}/data/processed/arxiv_articles.csv",
                         dtype={'article_id':str},parse_dates=['date'],
                         usecols=['article_id','abstract','created','date','is_ai'])
    
    logging.info("Loading categories")
    cats = pd.read_csv(f"{project_dir}/data/raw/arxiv_article_categories.csv",
                  dtype={'article_id':str})
    
    logging.info("Loading tokenised abstracts")
    with open(f"{project_dir}/data/interim/arxiv_tokenised.json",'r') as infile:
        arxiv_tokenised = json.load(infile)
    
    logging.info("Loading AI labelling outputs")
    with open(f"{project_dir}/data/interim/find_ai_outputs.p",'rb') as infile:
        ai_indices, term_counts = pickle.load(infile)    
    
    logging.info("Loading ArXiv category lookup")
    with open(f"{project_dir}/data/raw/arxiv_category_lookup.json",'r') as infile:
        arxiv_cat_lookup = json.load(infile)
    
    logging.info("Processing")
    papers['tokenised'] = papers['article_id'].map(arxiv_tokenised)
    
    # Create category sets to identify papers in different categories
    ai_cats = ['cs.AI','cs.NE','stat.ML','cs.LG']
    cat_sets = cats.groupby('category_id')['article_id'].apply(lambda x: set(x))

    # Create one hot encodings for AI categories
    ai_binary = pd.DataFrame(index=set(cats['article_id']),columns=ai_cats)

    for c in ai_binary.columns:
        ai_binary[c] = [x in cat_sets[c] for x in ai_binary.index]
    
    # Create arxiv dataset
    papers.set_index('article_id',inplace=True)
    
    # We remove papers without abstracts and arXiv categories
    arx = pd.concat([ai_binary,papers],axis=1,sort=True).dropna(
                                            axis=0,subset=['abstract','cs.AI'])
    
    return arx,ai_indices,term_counts,arxiv_cat_lookup,cat_sets,cats,ai_cats


def make_plot_1(arx,save=True):
    '''Makes first plot
    '''
    # First chart: trends
    ai_bool_lookup = {False:'Other categories',True:'AI'}

    # Totals
    ai_trends = arx.groupby(['date','is_ai']).size().reset_index(
        name='Number of papers')
    ai_trends['is_ai'] = ai_trends['is_ai'].map(ai_bool_lookup)

    # Shares
    ai_shares = ai_trends.pivot_table(
                                      index='date',
                                      columns='is_ai',
                                      values='Number of papers').fillna(
                                      0).reset_index(drop=False)
    ai_shares['share'] = ai_shares['AI']/ai_shares.sum(axis=1)

    #  Make chart
    at_ch = (alt.Chart(ai_trends).
             transform_window(roll = 'mean(Number of papers)',
                             frame=[-5,5],
                             groupby=['is_ai']).
             mark_line().
             encode(x=alt.X('date:T',title='',axis=alt.Axis(
                                                labels=False,ticks=False)),
                   y=alt.Y('roll:Q',title=['Number', 'of papers']),
                    color=alt.Color('is_ai:N',title='Category')).properties(
                                                width=350,
                                                height=120))
    as_ch = (alt.Chart(ai_shares).
            transform_window(roll = 'mean(share)',
                             frame=[-5,5]).
             mark_line()
             .encode(x=alt.X('date:T',title=''),
                    y=alt.Y('roll:Q',title=['AI as share', 'of all arXiv']))
             ).properties(width=350,height=120)

    ai_trends_chart = alt.vconcat(at_ch,as_ch,spacing=0)

    if save==True:
        save_altair(ai_trends_chart,'fig_1_ai_trends',driver=driv)

    return ai_trends_chart,ai_trends
    

# Read data
arx,ai_indices,term_counts,arxiv_cat_lookup,cat_sets,cats,ai_cats = load_process_data()

# Extract results
logging.info("AI descriptive statistics")
results = {}

# Q1: How many papers in total

ai_expanded = set(chain(*[x for x in ai_indices.values()]))
ai_core_dupes = list(chain(*[v for k,v in cat_sets.items() if k in ai_cats]))
ai_core = set(chain(*[v for k,v in cat_sets.items() if k in ai_cats]))
ai_new_expanded = ai_expanded - ai_core
ai_joint = ai_core.union(ai_expanded)

results['ai_expanded_n'] = len(ai_expanded)
results['ai_core_with_duplicates_n'] = len(ai_core_dupes)
results['ai_core_n'] = len(ai_core)
results['ai_new_expanded_n'] = len(ai_expanded - ai_core)
results['ai_joint'] =  len(ai_joint)

# Plot chart 1:
logging.info("Make first plot")
plot, trends = make_plot_1(arx)

# Cumulative analysis

# Get cumulative shares of activity
logging.info("Cumulative results")
ai_cumulative = ai_trends.pivot_table(
                                      index='date',
                                      columns='is_ai',
                                      values='Number of papers').fillna(
                                      0).apply(lambda x: x/x.sum()).cumsum()

paper_shares = ai_cumulative.loc[[x.to_pydatetime() in [datetime.datetime(2020,1,1),
                                                        datetime.datetime(2018,1,1),
                                                        datetime.datetime(2015,1,1),
                                                        datetime.date(2012,1,1)]
                   for x in ai_cumulative.index]]

# Add results
for rid,r in paper_shares.iterrows():
    results[f'Share of papers published before {str(rid.date())}']=100*np.round(
                                                                    r['AI'],2)






















with open(f"{project_dir}/reports/results.txt",'w') as outfile:
    for k,v in results.items():
        outfile.writelines(k+': '+str(v)+'\n')

