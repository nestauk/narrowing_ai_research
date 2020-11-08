import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from narrowing_ai_research.utils.list_utils import flatten_list
from matplotlib import cm
import matplotlib.patches as mpatches


def make_network_from_doc_term_matrix(mat, threshold, id_var):
    """Create a network from a document term matrix.
    Args
        Document term matrix where the rows are documents and the columns are topics
        threshold is the threshold to consider that a topic is present in a matrix.
    Returns:
        A network
    """
    # Melt the topic mix and remove empty entries
    cd = pd.melt(mat, id_vars=[id_var])

    cd = cd.loc[cd["value"] > threshold]

    # This gives us the topic co-occurrence matrix
    co_occurrence = cd.groupby(id_var)["variable"].apply(lambda x: list(x))

    # Here the idea is to create a proximity matrix based on co-occurrences

    # Turn co-occurrences into combinations of pairs we can use to construct a
    # similarity matrix
    sector_combs = flatten_list(
        [sorted(list(combinations(x, 2))) for x in co_occurrence]
    )
    sector_combs = [x for x in sector_combs if len(x) > 0]

    # Turn the sector combs into an edgelist
    edge_list = pd.DataFrame(sector_combs, columns=["source", "target"])

    edge_list["weight"] = 1

    # Group over edge pairs to aggregate weights
    edge_list_weighted = (
        edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
    )

    edge_list_weighted.sort_values("weight", ascending=False).head(n=10)

    # Create network and extract communities
    net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

    return net

def make_co_network(papers,topic_mix,topic_category_map,year_list):
    '''Extract co-occurrence network
    '''
    
    papers_sel = set(papers.loc[papers['year'].isin(year_list)]['article_id'])
    topic_mix_sel = topic_mix.loc[topic_mix['article_id'].isin(papers_sel)]
    
    # Extract the network
    net = make_network_from_doc_term_matrix(topic_mix_sel,
                                            threshold=0.1,id_var='article_id')
    
    nx.set_node_attributes(net,topic_category_map,'category')
    
    # Create size lookup
    size_lookup = topic_mix_sel.iloc[:,1:].applymap(
                                            lambda x: x>0.05).sum().to_dict()
    
    return net,size_lookup

def plot_network(net,size_distr,cat_list,palette='Accent'):
    '''Plot co-occurrence network
    '''
    
    pal = cm.get_cmap(palette)
    
    color_lookup = {name:pal(num) for num,name in enumerate(cat_list)}

    patches = [mpatches.Patch(facecolor=c, label=arxiv_cat_lookup[l],
                              edgecolor='black') for l,c in color_lookup.items()]
    
    fig,ax = plt.subplots(figsize=(18,8))

    #Show the network
    show_network(ax,net,norm=10,norm_2=0.7,color_lookup=color_lookup,
                 size_lookup=size_distr,
                 layout = nx.spring_layout,
                 #layout = nx.kamada_kawai_layout,
                 label='All',loc=(-0.5,1.48),ec='black',alpha=1)

    #Draw the legend
    ax.legend(handles=patches,facecolor='white',
              loc='lower center',title='Category',ncol=4)

    #Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

def plot_comp_network(nets,size_distrs,cat_list,
                      arxiv_cat_lookup,topic_category_map,
                      palette='Accent'):
    '''Plots two networks side by side
    '''
    pal = cm.get_cmap(palette)
    
    color_lookup = {name:pal(num) for num,name in enumerate(cat_list)}

    patches = [mpatches.Patch(facecolor=c, label=arxiv_cat_lookup[l],
                              edgecolor='black') for l,c in color_lookup.items()]
    
    fig,ax = plt.subplots(figsize=(14,16),nrows=2)
    
    for n in [0,1]:
        #Show the network
        show_network(ax[n],nets[n],
                     topic_category_map=topic_category_map,
                     norm=10,norm_2=0.7,color_lookup=color_lookup,
                     size_lookup=size_distrs[n],
                     layout = nx.spring_layout,
                     label='All',loc=(-0.5,1.48),ec='black',alpha=1)

        #Draw the legend
        ax[n].legend(handles=patches,
                     facecolor='white',
                     loc='lower center',
                     title='Category',ncol=3)

        #Remove ticks
        ax[n].set_xticks([])
        ax[n].set_yticks([])

def show_network(ax,net,
                 topic_category_map,
                 label,loc,         
                 size_lookup,
                 color_lookup,norm=2000,
                 norm_2=1.2,layout=nx.kamada_kawai_layout,
                 ec='white',alpha=0.6):
    '''
    Plots a network visualisation of a topic netwirk 
    '''
    
    new_net = net.copy()    
    new_net_2 = nx.maximum_spanning_tree(new_net)
    
    #Calculate the layout
    pos = layout(new_net_2,
                 center=(0.5,0.5)
                )
    
    node_s = list([size_lookup[x]**norm_2 for x in dict(new_net_2.degree).keys()])
    node_c = []
    
    for x in new_net_2.nodes:
        if x not in topic_category_map.keys():
            node_c.append('white')
        else:
            if topic_category_map[x] not in color_lookup.keys():
                node_c.append('white')
            else:
                c = color_lookup[topic_category_map[x]]
                node_c.append(c)
            
    #Draw the network. There is quite a lot of complexity here
    nx.draw_networkx_nodes(new_net_2,pos,
                       node_size=node_s,
                       node_color = node_c,
                       cmap='tab20c',
                       alpha=alpha,edgecolors='darkgrey',ax=ax)

    edge_w = [e[2]['weight']/norm for e in new_net_2.edges(data=True)]
    nx.draw_networkx_edges(new_net_2,pos,width=edge_w,
                           edge_color=ec,ax=ax,alpha=alpha)

def network_distance_stats(net,name):
    '''Extract various network statistics from a topic network
    '''
    # Extract large component
    large_comp = [nx.subgraph(net,n) for n in nx.connected_components(net)]
    
    # Number of components
    n_comps = len(large_comp)
        
    # Average path lengths in the large component
    av_sh_p_l = nx.average_shortest_path_length(large_comp[0],weight='weight')
    diam = nx.diameter(large_comp[0])
        
    return pd.Series([n_comps,av_sh_p_l,diam],name=
                     name,index=['components','average_path_length','diameter'])
        

