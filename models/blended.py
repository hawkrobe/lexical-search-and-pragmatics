import os
import walker
import warnings
import pragmatics

import pandas as pd
import numpy as np
import networkx as nx

from itertools import product
from joblib import Parallel, delayed

class blended:
  def __init__(self, exp_path):
    print("Init path:", os.path.abspath('.'))
    self.exp_path = exp_path

    # import target words
    target_df = pd.read_csv(f"{exp_path}/targets.csv")
    target_df["wordpair"] = target_df["Word1"]+ "-"+ target_df["Word2"]
    self.target_words = set(target_df.Word1).union(target_df.Word2)
    self.transitions = pd.read_csv(f'{exp_path}/model_input/swow_strengths.csv')\
                         .rename(columns={'R123.Strength' : 'weight'})
    self.biases = pragmatics.Selector(exp_path, alpha = 20, costweight = 0)\
                            .save_all_clues()

    # launch grid (note that this is very memory intensive)
    with Parallel(n_jobs=9) as parallel:
      parallel(
        delayed(self.save_candidates)(bias_weight, word1, word2)
        for (word1, word2), bias_weight
        in product(zip(target_df['Word1'], target_df['Word2']),
                   [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1])
      )

  def get_words_by_node(self, nodes):
    return [self.index_to_name[index] if index in self.index_to_name else None
            for index in nodes]

  def run_random_walks(self, bias_weight, clues):
    # instantiate graph
    G = nx.from_pandas_edgelist(self.transitions, 'cue', 'response', ['weight'], 
                                create_using=nx.DiGraph)

    # add bias for high fitness clues
    clue_bias = self.biases.query(f"targetpair == '{clues[0]}-{clues[1]}'")\
                         .rename(columns={'clueword' : 'response'})
    for s in G.copy().nodes: 
      for t, bias in zip(clue_bias['response'], clue_bias['raw_diagnosticity']):
        # add missing edges
        if not G.has_edge(s, t) and bias_weight * bias > 0.00001 :
          G.add_edge(s, t, weight= bias_weight * bias)

        # reweight edges
        elif G.has_edge(s, t):
          G[s][t]['weight'] = (1 - bias_weight) * G[s][t]['weight'] + bias_weight * bias

        # prune low-weight edges
        if G.has_edge(s, t) and G[s][t]['weight'] < 0.00001 :
          G.remove_edge(s, t)

    # run walks
    graph = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
    self.index_to_name = {k: v['word'] for k,v in graph.nodes(data=True)}
    self.name_to_index = {v['word'] : k for k,v in graph.nodes(data=True)}
    return walker.random_walks(
      graph, 
      n_walks=1000,
      walk_len=8195,
      start_nodes=[self.name_to_index[name] for name in self.target_words]
    )

  def save_candidates(self, bias_weight, word1, word2) :
    '''
    write out walks in order of words visited
    '''
    rw = self.run_random_walks(bias_weight, [word1, word2])

    print(f"Saving candidates for {word1}-{word2}-{bias_weight}")
    w1_walks = [x for x in rw if x[0] == self.name_to_index[word1]]
    w2_walks = [x for x in rw if x[0] == self.name_to_index[word2]]

    d = {f'walk-{int(2*i)}': self.get_words_by_node(w1_walks[i]) for i in range(1000)}
    d.update({f'walk-{int(2*i+1)}': self.get_words_by_node(w2_walks[i]) for i in range(1000)})
    
    # get cumulative sums
    df = pd.DataFrame(d)
    df['step'] = range(1, len(df) + 1)
    df = df.melt(id_vars='step', var_name='walk', value_name='Word')
    df['walk'] = df['walk'].str.replace('walk-', '')
    df['wordpair'] = word1 + '-' + word2
    df = df.groupby(['walk', 'Word', 'wordpair']).agg({'step': 'first'}).reset_index()
    df = df.groupby(['step', 'Word', 'wordpair']).size().reset_index(name='n')
    i = pd.MultiIndex.from_product([
      df['Word'].unique(), 
      df['wordpair'].unique(), 
      range(1, 8195)
    ], names=['Word', 'wordpair', 'step'])
    df = df.set_index(['Word', 'wordpair', 'step']) \
           .reindex(i, fill_value=0).reset_index()
    df = df.sort_values(['Word', 'wordpair', 'step'])
    df['cdf'] = df.groupby(['Word', 'wordpair'])['n'].cumsum() / 2000
    df = df[df['step'].isin(2 ** np.arange(14))]
    df['bias_weight'] = bias_weight

    output_path = os.path.join(self.exp_path, 'model_output', f'{word1}-{word2}-{bias_weight}-cdf-blended.csv')
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
  np.random.seed(12345)
  blended('../data/exp1')
