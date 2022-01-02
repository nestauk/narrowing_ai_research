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

# ## Analysis of influence

# +
# %load_ext autoreload
# %autoreload 2
# %config Completer.use_jedi = False
from narrowing_ai_research.utils.read_utils import read_papers, read_papers_orgs, read_topic_mix, paper_orgs_processing
import pandas as pd
import numpy as np
from toolz import pipe
from narrowing_ai_research import project_dir
import statsmodels.api as sm
from statsmodels.api import add_constant
from sklearn.decomposition import PCA
import altair as alt

from narrowing_ai_research.utils.altair_utils import altair_visualisation_setup, save_altair
# -

webd = altair_visualisation_setup()

# ### Read data

# +
papers = (read_papers(keep_vars=['article_id','year','date','is_ai','citation_count'])
          .query("is_ai == True")
          .reset_index(drop=True))

porgs = read_papers_orgs()

orgs = (paper_orgs_processing(porgs,papers)
        .query("is_ai==True")
        .reset_index(drop=True)
       )

tm = read_topic_mix()
# -

# ### Create analytical table

# +
# AI papers with private orgs

ai_comp = pipe(orgs.query("org_type=='Company'")['article_id'],set)
ai_num = orgs.groupby('article_id').size()
# -

papers_an = (papers
          .loc[papers['article_id'].isin(set(orgs['article_id']))]
          .query("is_ai == True")
          .assign(is_comp = lambda x: x['article_id'].isin(ai_comp))
          .assign(num_auth = lambda x: x['article_id'].map(ai_num))
          .reset_index(drop=True)
          .dropna())

(papers_an
 .groupby(['is_comp'])['citation_count']
 .agg(['mean','median','std','max'])
 .reset_index(drop=False).to_latex(f"{project_dir}/reports/citation_descr.tex",index=False))

tm.shape

# +
cit_evol_df= (papers_an
              .groupby(['is_comp','year'])['citation_count']
              .mean().unstack(level=0).dropna().loc[range(2012,2020)]
              .stack()
              .reset_index(name='mean_citations')
              .assign(is_comp = lambda df: df['is_comp'].replace({True:'Company',
                                                                  False:'Not company'}))
             )

cit_evol_chart = (alt.Chart(cit_evol_df)
                  .mark_line(point=True)
                  .encode(x=alt.X('year:O',title=None),
                          y=alt.Y('mean_citations',title='Mean citations'),
                          color=alt.Color('is_comp',title='Article type'))).properties(width=400,height=200)

save_altair(cit_evol_chart,"fig_influence",driver=webd)

# -

# ### Regression

# +
# Steps: train model, get model results

# +
def get_model_results(model, name):
    pass
def fit_model(papers_an,tm,comps_n):
    
    if comps_n == 0:
        reg_data = papers_an.copy()
        endog = reg_data["citation_count"].astype(float)
        exog = (add_constant(
            reg_data[["year","is_comp","num_auth"]])).astype(float)
    else:
        pca = PCA(n_components=comps_n)
        tm_pca = (pd.DataFrame(pca.fit_transform(tm.iloc[:,1:].dropna()))
                  .assign(article_id = tm['article_id']))
        
        tm_pca.columns = [str(x) for x in tm_pca]
        
        reg_data = papers_an.merge(tm_pca,
                            on='article_id')
        endog = reg_data["citation_count"].astype(float)
        
        exog = (add_constant(
            reg_data[["year","is_comp","num_auth"]+tm_pca.drop(axis=1,labels=['article_id'])
                         .columns.tolist()])
                  .astype(float))
    return sm.Poisson(endog=endog,exog=exog).fit_regularized(cov_type="HC1")
    
    
    
ms = [fit_model(papers_an,tm,n) for n in [0,10,50,100]]
# -

sg= Stargazer(ms)
sg.covariate_order(['const','is_comp','num_auth','year'])
sg.custom_columns(['No topics','PCA 10','PCA 50','PCA 100'],[1,1,1,1])
sg.show_adj_r2=False
sg.show_r2 = False
sg.show_dof = False
sg.show_f_statistic = False
sg.show_residual_std_err = False
sg.extract_data
sg.render_latex()
sg
print(sg.render_latex())


" & ".join([str(np.round(m.prsquared,3)) for m in ms])


# ### Analysis of collaborators

from numpy.random import choice

org_ids = orgs.groupby('article_id').apply(lambda df: 'Company' in set(df['org_type']))
corp_papers = pipe(orgs.loc[orgs['article_id'].isin(org_ids.loc[org_ids==True].index)]['article_id'],set)

orgs.loc[orgs['article_id'].isin(corp_papers)].groupby('article_id').apply(lambda df: 'Education' in set(df['org_type'])).mean()

corp_collabs = orgs.loc[orgs['article_id'].isin(corp_papers)].query("org_type!='Company'")['org_name'].value_counts()
corp_ranked = pd.Series(range(len(corp_collabs)),index=corp_collabs.index)

corp_quartile = pd.cut(corp_ranked,bins=[0,10,50,100,500,len(corp_collabs)],labels=range(5)[::-1],include_lowest=True)

paper_rank_collab = (orgs
                     .assign(collab_rank = lambda df: df['org_name'].map(corp_quartile))
                     .dropna(axis=0,subset=['collab_rank'])
                     [['article_id','collab_rank']]
                    )

# +
collab_ranks = (papers.merge(paper_rank_collab,
             on=['article_id'],how='left')
 .groupby('collab_rank')['citation_count'].mean()).to_frame().reset_index(drop=False)

#collab_r = {r:", ".join(choice(corp_quartile.loc[corp_quartile==r].index,5,replace=False)) for r in range(5)}

collab_r = {r:", ".join(corp_quartile.loc[corp_quartile==r].index[:5]) for r in range(5)}

collab_ranks['Sample organisations'] = collab_ranks['collab_rank'].map(collab_r)
collab_ranks['citation_count'] = [np.round(x,3) for x in collab_ranks['citation_count']]

collab_clean = {4: "1-10",3:"11-50",2:"50-100",1:"100-500",0:"Above 500"}

collab_ranks['collab_rank'] = collab_ranks['collab_rank'].map(collab_clean)
# -

with pd.option_context("max_colwidth", 1000):
    collab_ranks.to_latex(f"{project_dir}/reports/collaboration_table.tex",index=False)

collab_ranks


