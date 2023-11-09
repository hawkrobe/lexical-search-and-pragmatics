import random
from random import randrange
import os
import json
import pickle
import itertools
import walker
import math
import warnings
import scipy.spatial.distance as distance
import scipy
from heapq import nlargest
import pandas as pd
import numpy as np
import networkx as nx
import math
from collections import defaultdict
import difflib
import gensim.downloader as api
import time



warnings.simplefilter(action='ignore', category=FutureWarning)

class SWOW:
  '''
  Run random walks on the small world of words dataset of association norms
  '''

  def __init__(self, data_path):
    '''
    Initializes specific targets. Loads graph and random walks.

    Args:
      data_path: path to csv containing target words.  
    '''
    
    print("Init path:", os.path.abspath('.'))

    # import target words
    self.target_df = pd.read_csv(f"{data_path}/targets.csv".format())
    self.target_df["wordpair"]= self.target_df["Word1"]+ "-"+self.target_df["Word2"]
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)
    self.vocab = list(pd.read_csv("../data/vocab.csv").Word)
    self.load_graph(data_path)
    self.index_to_name = {k: v['word'] for k,v in self.graph.nodes(data=True)}
    self.name_to_index = {v['word'] : k for k,v in self.graph.nodes(data=True)}
    self.load_random_walks(data_path)

  def load_graph(self, data_path):
    '''
    Loads graph from reading in networkx graph saved as pickle, or creates new graph.
    '''
    if os.path.exists(f'{data_path}/walk_data/swow.gpickle') :
      with open(f'{data_path}/walk_data/swow.gpickle', 'rb') as f:
        self.graph = pickle.load(f)
    else :
      self.save_graph(data_path, None)

  def save_graph(self, path, threshold):
    '''
    Creates graph directly from pandas edge list and saves to file.
    '''
    path = path + 'walk_data/swow-strengths.csv'
    edges = pd.read_csv(path).rename(columns={'R123.Strength' : 'weight'})
    G = nx.from_pandas_edgelist(edges, 'cue', 'response', ['weight'], create_using=nx.DiGraph)
    G = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
    with open('../data/walk_data/swow.gpickle', 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

  def load_random_walks(self, data_path):
    '''
    Loads saved random walks from pickle, or run new walks. 
    '''
    if os.path.exists(f'{data_path}/walk_data/walks.pkl'):
      with open(f'{data_path}/walk_data/walks.pkl', 'rb') as f:
        self.rw = pickle.load(f)
    else :
      self.save_random_walks()

  def save_random_walks(self, n_walks = 1000, walk_len = 10000):
    '''
    Runs n_walks independent random walks of walk_len length from each words.
    '''
    indices = self.get_nodes_by_word(self.target_words)
    self.rw = walker.random_walks(self.graph, n_walks=n_walks, walk_len=walk_len, start_nodes=indices, alpha = 0.1)

    with open('../data/walk_data/walks.pkl', 'wb') as f:
      pickle.dump(self.rw, f)

  def save_candidates(self) :
    for word1, word2 in zip(self.target_df['Word1'], self.target_df['Word2']) :
      w1_walks = np.array([x for x in self.rw if x[0] == self.get_nodes_by_word([word1])])
      w2_walks = np.array([x for x in self.rw if x[0] == self.get_nodes_by_word([word2])])
      d = {f'walk-{int(2*i)}': self.get_words_by_node(w1_walks[i]) for i in range(10)}
      d.update({f'walk-{int(2*i+1)}': self.get_words_by_node(w2_walks[i]) for i in range(10)})
      with open(f'../data/{word1}-{word2}-walks.json', 'w', encoding ='utf8') as json_file:
        json.dump(d, json_file, ensure_ascii = False)

  def chunk(self, l, n):
    '''
    iterates through l in chunks of size n
    '''
    c = itertools.count()
    return (list(it) for _, it in itertools.groupby(l, lambda x: next(c)//n))

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

  def union_intersection_candidates(self, w1, w2):
    '''
    Compares two walks for a given word pair, finding union and intersection of candidate words in both paths
    for path lengths budgeted from 2 - 2^10. Returns candidate words with likelihoods of being visited.

    Args:
      w1, w2: target words.
    Returns:
      union_counts, intersection_count : dict of  { budget : {wordid:weight} }    
    '''

    # Retrieve paths that start with target word
    target_indices = self.get_nodes_by_word([w1, w2])
    w1_walks = np.array([x for x in self.rw if x[0] == target_indices[0]]).tolist()
    w2_walks = np.array([x for x in self.rw if x[0] == target_indices[1]]).tolist()

    union_avg = {budget: defaultdict(list) for budget in self.powers_of_two(10000)}
    intersect_avg = {budget: defaultdict(list) for budget in self.powers_of_two(10000)}

    # Count union/intersection appearances
    for w1_walk, w2_walk in zip(w1_walks, w2_walks) :
      union_counts = {budget: defaultdict(int) for budget in self.powers_of_two(10000)}
      intersect_counts =  {budget: defaultdict(int) for budget in self.powers_of_two(10000)}
      for search_budget in self.powers_of_two(10000) :
        intersect = set(w1_walk[: search_budget]).intersection(w2_walk[: search_budget])
        union = set(w1_walk[: search_budget]).union(w2_walk[: search_budget])
        for node in range(len(self.vocab)) :
          intersect_avg[search_budget][node] += [(1/len(intersect) + 0.000001) if node in intersect else 0.0000001]
          union_avg[search_budget][node] += [(1/len(union) + 0.000001) if node in union else 0.0000001]

    return ({k: {element : np.mean(l) for (element, l) in d.items()} for (k, d) in union_avg.items()},
            {k: {element : np.mean(l) for (element, l) in d.items()} for (k, d) in intersect_avg.items()})


  def clue_score(self, clues, w1, w2): 
    '''
    Extract info for given clue and word-pair.

    Args: 
      w1, w2: word pair
      clues: array of clues generated from word pair  
    Returns: 
      (union score, intersection score) of word
    '''
    clue_indices = self.get_nodes_by_word(clues)
    union_counts, intersection_counts = self.union_intersection_candidates(w1, w2)
    return ({budget: [d[clue_index] for clue_index in clue_indices] for (budget, d) in union_counts.items()},
            {budget: [d[clue_index] for clue_index in clue_indices] for (budget, d) in intersection_counts.items()})

  def save_scores(self, data_path):
    '''
    Computes and saves clue scores to scores.csv
    '''
    # import empirical clues (cleaned)
    expdata = pd.read_csv(f"{data_path}", encoding= 'unicode_escape')
    scores = defaultdict(list)
    rows = []
    # look up how often each clue was visited
    for name, group in expdata.groupby('wordpair') :
      union_score, intersect_score = self.clue_score(
        group['correctedClue'].to_numpy(),
        group['Word1'].to_numpy()[0],
        group['Word2'].to_numpy()[0]
      )
      for key in union_score.keys() :
        scores['union_' + str(key)].extend(union_score[key])
        scores['intersection_' + str(key)].extend(intersect_score[key])
      rows.append(group)

    # save to file
    pd.concat(
      [
        pd.concat(rows,axis=0,ignore_index=True),
        pd.DataFrame.from_dict(scores)
      ],
      axis=1
    ).to_csv('/'.join(data_path.split('/')[:-1])+'/scores.csv')

  def save_rank_order(self, data_path, permute = False):
    '''
    Tracks of the number of times a word is visited for different budgets,
    across all word pairs’ walks.
    '''

    # Loop through word pairs
    expdata = pd.read_csv(f"{data_path}", encoding= 'unicode_escape')
    if permute :
      expdata['correctedClue'] = expdata['correctedClue'].sample(frac=1).values

    out = []
    for name, group in expdata.groupby('wordpair') :
      print(name)
      # convert words to nodes
      target_nodes = self.get_nodes_by_word([group['Word1'].to_numpy()[0], group['Word2'].to_numpy()[0]])
      clue_nodes = self.get_nodes_by_word(group['correctedClue'].to_numpy())

      # loop through 10000 pairs of walks to get indices of first appearances
      w1_walks = np.array([x for x in self.rw if x[0] == target_nodes[0]]).tolist()
      w2_walks = np.array([x for x in self.rw if x[0] == target_nodes[1]]).tolist()
      w1_walks_ord = [list(np.unique(walk)[np.argsort(np.unique(walk, return_index=True)[1])]) for walk in w1_walks]
      w2_walks_ord = [list(np.unique(walk)[np.argsort(np.unique(walk, return_index=True)[1])]) for walk in w2_walks]
      new_cols = defaultdict(list)
      for w1_walk, w2_walk in zip(w1_walks_ord, w2_walks_ord) :
        new_cols[f'w1_index_walk{len(new_cols.keys())}'] = [w1_walk.index(clue_node) if clue_node in w1_walk else 10000 for clue_node in clue_nodes]
        new_cols[f'w2_index_walk{(len(new_cols.keys()) - 1)}'] = [w2_walk.index(clue_node) if clue_node in w2_walk else 10000 for clue_node in clue_nodes]
      out.append(pd.concat(
        [group.reset_index(), pd.DataFrame.from_dict(new_cols)],
        axis = 1
      ))

    pd.concat(out).to_csv(
      f'../data/exp1/model_output/indices{"_permuted" if permute else ""}.csv'
    )

if __name__ == "__main__":
  swow = SWOW('../data')
  np.random.seed(444)
#  swow.save_candidates()
  swow.save_rank_order('../data/exp1/exp1-cleaned.csv', permute = False)
  swow.save_rank_order('../data/exp1/exp1-cleaned.csv', permute = True)
