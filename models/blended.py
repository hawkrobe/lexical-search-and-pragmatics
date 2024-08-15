import os
import walker
import warnings

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
    self.target_df = pd.read_csv(f"{exp_path}/targets.csv")
    self.target_df["wordpair"]= self.target_df["Word1"]+ "-"+self.target_df["Word2"]
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)
    self.vocab = pd.read_csv(f"{exp_path}/model_input/vocab.csv")
    self.vocab_size = len(list(self.vocab.Word))
    self.transitions = pd.read_csv(f'{exp_path}/model_input/swow_strengths.csv')\
                         .rename(columns={'R123.Strength' : 'weight'}) 

    # generated with "python pragmatics.py cdf RSA 100 0"
    self.sims = pd.read_csv(f"{exp_path}/model_output/speaker_df_allclues.csv")

    # launch grid
    with Parallel(n_jobs=2) as parallel:
      parallel(
        delayed(self.save_candidates)(exp_path, cost_weight, word1, word2)
        for (word1, word2), cost_weight
        in product(zip(self.target_df['Word1'], self.target_df['Word2']), 
                   [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
      )

  def get_words_by_node(self, nodes):
    return [self.index_to_name[index] if index in self.index_to_name else None
            for index in nodes]

  def run_random_walks(self, cost_weight, clues):
    # instantiate graph
    G = nx.from_pandas_edgelist(self.transitions, 'cue', 'response', ['weight'], 
                                create_using=nx.DiGraph)
    
    # add bias for high fitness clues
    clue_bias = self.sims.query(f"targetpair == '{clues[0]}-{clues[1]}'")\
                         .rename(columns={'clueword' : 'response'})
    
    for s in G.nodes: 
      for t, bias in zip(clue_bias['response'], clue_bias['prob']):
        # add missing edges
        if not G.has_edge(s, t) and bias > 0.001 :
          G.add_edge(s, t, weight=bias)
        # reweight edge
        if G.has_edge(s, t) :
          w = cost_weight * G[s][t]['weight'] + (1-cost_weight) * bias
          G[s][t]['weight'] = w
        # prune low-weight edges
        if G.has_edge(s, t) and G[s][t]['weight'] < 0.001 :
          G.remove_edge(s, t)

    # run walks
    self.graph = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
    self.index_to_name = {k: v['word'] for k,v in self.graph.nodes(data=True)}
    self.name_to_index = {v['word'] : k for k,v in self.graph.nodes(data=True)}
    self.rw = walker.random_walks(
      self.graph, 
      n_walks=1000, 
      walk_len=8195,
      start_nodes=[self.name_to_index[name] for name in self.target_words]
    )

  def save_candidates(self, exp_path, cost_weight, word1, word2) :
    '''
    write out walks in order of words visited
    '''
    self.run_random_walks(cost_weight, [word1, word2])
    print(f"Saving candidates for {word1}-{word2}-{cost_weight}")
    w1_walks = [x for x in self.rw if x[0] == self.name_to_index[word1]]
    w2_walks = [x for x in self.rw if x[0] == self.name_to_index[word2]]
  
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
    df = df[df['step'].isin( 2 ** np.arange(14))]
    df['cost_weight'] = cost_weight

    output_path = os.path.join(exp_path, 'model_output', f'{word1}-{word2}-{cost_weight}-cdf-blended.csv')
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
  np.random.seed(1234)
  swow_exp1 = blended('../data/exp1')
  # swow_exp2 = blended('../data/exp2')
  # swow_exp3 = blended('../data/exp3')
