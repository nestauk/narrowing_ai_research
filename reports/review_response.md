# A Narrowing of AI research? Response to reviewers

First of all, we would like to thank our reviewers for their insightful comments and suggestions, which we have sought to address comprehensively. We believe that doing this has incresased significantly the robustness, contribution and policy-relevance of the paper while reducing its length. We go through these changes in turn.

1. We have reorganised and tightened the literature review so that its structure follows more closely Table 1, as reviewer 2 suggested, and our presentation of findings. This involves creating subsections about the value of technological diversity, economic forces that might narrow it down and the role of the private sector in these processes. In each of its subsections, we have integrated the economics/management/innovation studies literature with AI-focused research using examples and debates from the latter to ground the former.
2. We have moved to a technical annex the detailed description of the procedure we use to identify AI papers, our analysis of the topic co-occurrence network and a new section where we explore an expanded definition of AI that includes citations to/from our original corpus inspired by comments from Reviewer 2 (more on that below)
3. We have reorganised the findings section to make it flow better. The section consists of three sections now. The first one is on trends and includes AI research trends and the evolution of thematic diversity, the second one focuses on the comparison between public and private AI research, and the third (new) one studies the influence of private AI research (more on the last one below).
4. We have removed a substantial amount of repetitive text as well as some parts of the analysis that contributed less to the narrative, such as the analysis of the evolution of topic shares and their centrality, the comparison between thematic diversity of public and private research using total levels of activity, which was superseded by our analysis using corpora of the same size, and the semantic map based on sentence embeddings. 
5. We have included a robustness section that addresses several comments about the methodology, particularly from reviewer 2. This includes assessing the robustness of our methodology to an alternative definition of AI that includes citations to / from a sample of our corpus, training an alternative LDA topic model using a variable number of topics, and analysing a time adjusted corpus where we sample the same number of papers every year to avoid biases from the concentration of AI research in recent years. 
6. This section draws on a new analysis included in the technical annex where we extract citations to / from a sample of our corpus and study the extent to which they were captured in our original AI corpus. We also compare the thematic composition of new papers identified this way which were not part of the original corpus. This reveals a number of papers outside of our corpus which, as Reviewer 3 speculated, capture applications of AI in other domains. We include those papers in our robustness tests to assess if they change our findings.
7. We have included a new section of findings that focuses on the influence of private AI companies through an analysis of the number of citations they receive, and their collaboration of other organisations. This new section addresses an important point made by reviewer 2 that private sector companies comprise the minority of the corpus and therefore we would not expect their activities to explain the diversity trends presented in the paper. By showing that papers by private companies tend to be more highly cited, and that they often involve collaborations with prestigious institutions, we provide evidence supporting the idea that they may be influencing indirectly the trajectory of AI research in a way that makes it narrower. This new analysis connects with recent discussions in the AI community about a 'capture' of academic researchers by industry that we did not address in the first version of the paper, and which believe strenghtens our case and makes it more policy-relevant.
9. In the literature review, we have removed the confusion between directional / directed technical change pointed out by both reviewers. We have also followed Reviewer 3's suggestions by discussing our level of analysis (what do we mean by an AI technology) connected our analysis with the product life-cycle literature more explicity, and provided a rationale for why private sector companies might be so active in basic / open AI research.
10. In the data and methodology section we have created a subsection that specifically discusses potential sources of bias created by our choice of sources and the matching procedures we use. This picks up on some comments by reviewer 2 about potential biases created by non-disclosure of research results by firms, and the exclusion of smaller companies that are not captured by MAG / GRID. We have also sought to clarify the description of the topSBM algorithm and its advantages, and removed the inaccurate implication that this technique somehow estimates an "optimal" level of topics. We have taken onboard Reviewer 2 concerns about our reliance on a black box method by testing the robustness of our approach to an alternative modelling strategy using LDA, with findings that we report in the robustness section.
11. We have removed unnecessary mathematical notation that as Reviewer 2 pointed out was more confusing than enlightening in many occasions.
12. We have removed all language that could suggest our analysis is generating evidence about causal drivers, or that we are testing hypotheses from a formal model. We are clear that we are presenting a descriptive analysis and that we hope that future work will provide causal evidence about the impact of private sector activities on the diversity of AI research.
13.  We hope that our analysis of robustness of findings using an alternative corpus that includes additional papers identified through a citation analysis contributes to addressing Reviewer 2's comments about our reliance on heuristically selected parameteres to identify AI papers. Optimally, we would have re-run all our pipeline using an alternative set of parameters chosen based on their position within the distribution instead of hard values as reviewer 2 suggests but it would have been computationally expensive to re-run topSBM on the whole corpus using alternative parametrisatiosn of the pipeline so we opted for testing the robustness of the pipeline wholesale with an alternative approach. We would however be happy to re-run all the analysis (perhaps using a sample of the corpus) if the reviewers believe it is important.

Together, these changes have allowed us to reduce the size of the paper from 50 pages to 34 (38 including citations). We have kept the technical annex at the end of the paper to facilitate its review for now.

We would like to conclude by thanking the reviewers for their patience thoroughly reviewing this long paper, and the insights of their comments and suggestions, which we believe have materially improved its quality.