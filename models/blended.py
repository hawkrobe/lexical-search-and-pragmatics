import os
import walker

import pandas as pd
import numpy as np
import networkx as nx

from pragmatics import Selector
from itertools import product
from joblib import Parallel, delayed
from scipy.special import softmax

class blended:
  def __init__(self, exp_path):
    print("Init path:", os.path.abspath('.'))
    self.exp_path = exp_path

    # import target words
    self.target_df = pd.read_csv(f"{exp_path}/targets.csv")
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)
    self.vocab = pd.read_csv(f"{exp_path}/model_input/vocab.csv")
    self.vocab_size = len(list(self.vocab.Word))
    self.cluedata = pd.read_csv(f"{exp_path}/cleaned.csv")

    # expand to all rows
    self.transitions = pd.read_csv(f'{exp_path}/model_input/swow_strengths.csv')\
                         .rename(columns={'R123.Strength' : 'weight'})\
                         .set_index(['cue','response']).unstack(fill_value=0.0001).stack()\
                         .reset_index()
    self.selector = Selector(exp_path)

    # launch grid
    with Parallel(n_jobs=100) as parallel:
      parallel(
        delayed(self.save_candidates)(exp_path, row, cost_weight, dist_weight)
        for ((i, row), cost_weight, dist_weight)
        in product(self.target_df.iterrows(),
                   [0, 1, 2, 4, 8, 16, 100],
                   [0, 1, 2, 4, 8, 16, 100])
      )

  def get_words_by_node(self, nodes):
    return [self.index_to_name[index] 
            if index in self.index_to_name else None
            for index in nodes]

  def run_random_walks(self, row, cost_weight, dist_weight):
    # instantiate graph
    boardname = row['boardnames']
    wordpair = row['wordpair']
    inf = self.selector.fit(boardname, wordpair)
    biases = pd.DataFrame({'response' : self.vocab['Word'], 'bias' : inf})
    edges = pd.merge(self.transitions, biases)
    edges['combination'] = cost_weight * np.log(edges['weight']) \
                           + dist_weight * np.log(edges['bias'])
    edges['prob'] = edges.groupby(['cue'])['combination'].transform(softmax)

    # sparsify
    edges = edges[edges['prob'] > 0.0001]
    G = nx.from_pandas_edgelist(edges, 'cue', 'response', ['prob'], 
                                create_using=nx.DiGraph)
    
    # make sure all nodes are in there
    for s in self.vocab['Word']: 
      if not G.has_node(s) :
          G.add_node(s)

    # run walks
    self.graph = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
    self.index_to_name = {k: v['word'] for k,v in self.graph.nodes(data=True)}
    self.name_to_index = {v['word'] : k for k,v in self.graph.nodes(data=True)}
    self.rw = walker.random_walks(
      self.graph, 
      n_walks=1000, 
      walk_len=1024, 
      start_nodes=[self.name_to_index[name] for name in wordpair.split('-')]
    )

  def save_candidates(self, exp_path, row, cost_weight, dist_weight) :
    '''
    write out walks in order of words visited
    '''
    self.run_random_walks(row, cost_weight, dist_weight)
    print(f"Saving candidates for {row['wordpair']}-{cost_weight}-{dist_weight}")
    word1, word2 = row['wordpair'].split('-')
    w1_walks = [x for x in self.rw if x[0] == self.name_to_index[word1]]
    w2_walks = [x for x in self.rw if x[0] == self.name_to_index[word2]]
  
    d = {f'walk-{int(2*i)}': self.get_words_by_node(w1_walks[i]) for i in range(1000)}
    d.update({f'walk-{int(2*i+1)}': self.get_words_by_node(w2_walks[i]) for i in range(1000)})
    
    # get cumulative sums
    df = pd.DataFrame(d)
    df['step'] = range(1, len(df) + 1)
    df = df.melt(id_vars='step', var_name='walk', value_name='Word')
    df['walk'] = df['walk'].str.replace('walk-', '')
    df['wordpair'] = row['wordpair']
    df = df.groupby(['walk', 'Word', 'wordpair']).agg({'step': 'first'}).reset_index()
    df = df.groupby(['step', 'Word', 'wordpair']).size().reset_index(name='n')
    i = pd.MultiIndex.from_product([
      df['Word'].unique(), 
      df['wordpair'].unique(), 
      range(1, 1025)
    ], names=['Word', 'wordpair', 'step'])
    df = df.set_index(['Word', 'wordpair', 'step']) \
           .reindex(i, fill_value=0).reset_index()
    df = df.sort_values(['Word', 'wordpair', 'step'])
    df['cdf'] = df.groupby(['Word', 'wordpair'])['n'].cumsum() / 2000
    df = df[df['step'].isin( 2 ** np.arange(14))]
    df['cost_weight'] = cost_weight
    df['dist_weight'] = dist_weight

    # take softmax over words at each step
    df['prob'] = df.groupby(['wordpair', 'step'])["cdf"].transform(lambda x: softmax(np.log(x)))
    df = df[df['Word'].isin(self.cluedata['correctedClue'])]

    output_path = os.path.join(exp_path, 'model_output', f'{row['wordpair']}-{cost_weight}-{dist_weight}-cdf-blended.csv')
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
  np.random.seed(1235)
  swow_exp1 = blended('../data/exp1')
  # swow_exp2 = blended('../data/exp2')
  # swow_exp3 = blended('../data/exp3')
