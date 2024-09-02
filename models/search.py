import random
import os
import json
import pickle
import itertools
import walker
import math
import warnings

import pandas as pd
import numpy as np
import networkx as nx

from collections import defaultdict, Counter
from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=FutureWarning)

class SWOW:
  '''
  Run random walks on the small world of words dataset of association norms
  '''

  def __init__(self, exp_path):
    '''
    Initializes specific targets. Loads graph and random walks.

    Args:
      exp_path: path to csv containing target words.
    '''
    
    print("Init path:", os.path.abspath('.'))

    # import target words
    self.target_df = pd.read_csv(f"{exp_path}/targets.csv")
    self.target_df["wordpair"]= self.target_df["Word1"]+ "-"+self.target_df["Word2"]
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)
    self.vocab = pd.read_csv(f"{exp_path}/model_input/vocab.csv")
    self.vocab_size = len(list(self.vocab.Word))
    self.construct_graph(exp_path)
    self.index_to_name = {k: v['word'] for k,v in self.graph.nodes(data=True)}
    self.name_to_index = {v['word'] : k for k,v in self.graph.nodes(data=True)}
    self.run_random_walks(exp_path)

  def construct_graph(self, exp_path):
    '''
    Loads graph from reading in networkx graph saved as pickle, or creates new graph.
    '''
    if os.path.exists(f'{exp_path}/model_input/swow.gpickle') :
      with open(f'{exp_path}/model_input/swow.gpickle', 'rb') as f:
        self.graph = pickle.load(f)
    else :
      edges = pd.read_csv(f'{exp_path}/model_input/swow_strengths.csv').rename(columns={'R123.Strength' : 'weight'})
      G = nx.from_pandas_edgelist(edges, 'cue', 'response', ['weight'], create_using=nx.DiGraph)
      self.graph = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
      with open(f'{exp_path}/model_input/swow.gpickle', 'wb') as f:
        pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

  def run_random_walks(self, exp_path, n_walks = 1000, walk_len = 10000):
    '''
    Loads saved random walks from pickle, or run new walks. 
    '''
    if os.path.exists(f'{exp_path}/model_input/walks.pkl'):
      with open(f'{exp_path}/model_input/walks.pkl', 'rb') as f:
        self.rw = pickle.load(f)
    else :
      indices = self.get_nodes_by_word(self.target_words)
      self.rw = walker.random_walks(self.graph, n_walks=n_walks, walk_len=walk_len, start_nodes=indices, alpha = 0)
      with open(f'{exp_path}/model_input/walks.pkl', 'wb') as f:
        pickle.dump(self.rw, f)

  def powers_of_two(self, n):
    '''
    returns all powers of 2 between 2 and n
    '''
    return [2**i for i in range(1, int(math.log(n, 2))+1)]

  def get_nodes_by_word(self, words):
    '''
    looks up which node IDs a list of words correspond to
    '''
    return [self.name_to_index[name] if name in self.name_to_index else None
            for name in words]

  def get_words_by_node(self, nodes):
    '''
    looks up which words a list of node IDs correspond to
    '''
    return [self.index_to_name[index] if index in self.index_to_name else None
            for index in nodes]

  def save_candidates(self, exp_path) :
    '''
    write out walks in order of words visited
    '''
    for word1, word2 in zip(self.target_df['Word1'], self.target_df['Word2']) :
      print(word1, word2)
      w1_walks = np.array([x for x in self.rw if x[0] == self.get_nodes_by_word([word1])])
      w2_walks = np.array([x for x in self.rw if x[0] == self.get_nodes_by_word([word2])])
      d = {f'walk-{int(2*i)}': self.get_words_by_node(w1_walks[i]) for i in range(1000)}
      d.update({f'walk-{int(2*i+1)}': self.get_words_by_node(w2_walks[i]) for i in range(1000)})
      with open(f'{exp_path}/model_output/walks/{word1}-{word2}-walks.json', 'w', encoding ='utf8') as json_file:
        json.dump(d, json_file, ensure_ascii = False)

if __name__ == "__main__":
  np.random.seed(1235)
  swow_exp4 = SWOW('../data/exp4')
  swow_exp4.save_candidates('../data/exp4')
