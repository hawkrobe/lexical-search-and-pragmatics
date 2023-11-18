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
    self.target_df = pd.read_csv(f"{exp_path}/model_input/targets.csv")
    self.target_df["wordpair"]= self.target_df["Word1"]+ "-"+self.target_df["Word2"]
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)
    self.vocab = pd.read_csv(f"{exp_path}/model_input/vocab.csv")
    self.vocab_size = list(self.vocab.Word)
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
      edges = pd.read_csv(f'{exp_path}/model_input/swow-strengths.csv').rename(columns={'R123.Strength' : 'weight'})
      G = nx.from_pandas_edgelist(edges, 'cue', 'response', ['weight'], create_using=nx.DiGraph)
      G = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
      with open(f'{model_input}/model_input/swow.gpickle', 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

  def run_random_walks(self, exp_path):
    '''
    Loads saved random walks from pickle, or run new walks. 
    '''
    if os.path.exists(f'{exp_path}/model_input/walks.pkl'):
      with open(f'{exp_path}/model_input/walks.pkl', 'rb') as f:
        self.rw = pickle.load(f)
    else :
      indices = self.get_nodes_by_word(self.target_words)
      self.rw = walker.random_walks(self.graph, n_walks=n_walks, walk_len=walk_len, start_nodes=indices, alpha = 0.1)
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
      w1_walks = np.array([x for x in self.rw if x[0] == self.get_nodes_by_word([word1])])
      w2_walks = np.array([x for x in self.rw if x[0] == self.get_nodes_by_word([word2])])
      d = {f'walk-{int(2*i)}': self.get_words_by_node(w1_walks[i]) for i in range(1000)}
      d.update({f'walk-{int(2*i+1)}': self.get_words_by_node(w2_walks[i]) for i in range(1000)})
      with open(f'{exp_path}/model_output/{word1}-{word2}-walks.json', 'w', encoding ='utf8') as json_file:
        json.dump(d, json_file, ensure_ascii = False)


  def save_scores(self, exp_path, permute = False):
    '''
    Computes and saves clue scores to scores.csv
    '''

    expdata = pd.read_csv(f"{exp_path}/clean.csv", encoding= 'unicode_escape')
    if permute :
      expdata['correctedClue'] = expdata['correctedClue'].sample(frac=1).values

    scores = defaultdict(list)
    rows = []

    for name, group in expdata.groupby('wordpair') :
      print(name)
      # look up how often each clue was visited
      clue_indices = self.get_nodes_by_word(group['correctedClue'].to_numpy())
      w1 = group['Word1'].to_numpy()[0]
      w2 = group['Word2'].to_numpy()[0]
      target_indices = self.get_nodes_by_word([w1, w2])
      w1_walks = np.array([x for x in self.rw if x[0] == target_indices[0]]).tolist()
      w2_walks = np.array([x for x in self.rw if x[0] == target_indices[1]]).tolist()
      freq = dict(zip(self.vocab["Word"], 10**freqs["LgSUBTLWF"]))

      union_avg = {budget: defaultdict(list) for budget in self.powers_of_two(10000)}
      intersect_avg = {budget: defaultdict(list) for budget in self.powers_of_two(10000)}
      w1_avg = {budget: defaultdict(list) for budget in self.powers_of_two(10000)}
      w2_avg = {budget: defaultdict(list) for budget in self.powers_of_two(10000)}
      freq_avg = {budget: defaultdict(list) for budget in self.powers_of_two(10000)}

      # Count union/intersection appearances
      for w1_walk, w2_walk in zip(w1_walks, w2_walks) :
        for search_budget in self.powers_of_two(10000) :
          w1_counts = Counter(w1_walk[: search_budget])
          w2_counts = Counter(w2_walk[: search_budget])
          intersect = w1_counts & w2_counts
          union = w1_counts | w2_counts
          for node in range(len(self.vocab_size)) :
            intersect_avg[search_budget][node] += [(intersect[node]/(intersect.total() + 1) + 0.000001) if node in intersect else 0.0000001]
            union_avg[search_budget][node] += [(union[node]/(union.total() + 1) + 0.000001) if node in union else 0.0000001]
            w1_avg[search_budget][node] += [(w1_counts[node]/(w1_counts.total() + 1) + 0.000001) if node in w1_counts else 0.0000001]
            w2_avg[search_budget][node] += [(w2_counts[node]/(w2_counts.total() + 1) + 0.000001) if node in w2_counts else 0.0000001]
            freq_avg[search_budget][node] += [freq[node] if node in freq.keys() else 0.000001]

      # aggregate
      for key in union_counts.keys() :
        for i in clue_indices :
          scores[f'union_{str(key)}'] += [np.mean(union_avg[key][i]) if i != None else None]
          scores[f'intersection_{str(key)}'] += [np.mean(intersect_avg[key][i]) if i != None else None]
          scores[f'w1_{str(key)}'] += [np.mean(w1_avg[key][i]) if i != None else None]
          scores[f'w2_{str(key)}'] += [np.mean(w2_avg[key][i]) if i != None else None]
          scores[f'freq_{str(key)}'] += [np.mean(freq_avg[key][i]) if i != None else None]
      rows.append(group)

    # save to file
    pd.concat(
      [
        pd.concat(rows,axis=0,ignore_index=True),
        pd.DataFrame.from_dict(scores)
      ],
      axis=1
    ).to_csv(
      f'{exp_path}/model_output/scores{"_permuted" if permute else ""}.csv'
    )

  def save_rank_order(self, exp_path, permute = False):
    '''
    Tracks of the number of times a word is visited for different budgets,
    across all word pairs’ walks.
    '''

    # Loop through word pairs
    expdata = pd.read_csv(f"{exp_path}/clean.csv", encoding= 'unicode_escape')
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
      intersections = []
      unions = []
      for (walk1,walk2) in zip(w1_walks_ord, w2_walks_ord) :
        intersection = []
        union = []
        for word1, word2 in zip(walk1, walk2) :
          if word1 not in walk2 or walk2.index(word1) >= walk1.index(word1) :
            union.append(word1)
          elif walk2.index(word1) <= walk1.index(word1) :
            intersection.append(word1)
          if word2 not in walk1 or walk1.index(word2) >= walk2.index(word2) :
            union.append(word2)
          elif walk1.index(word2) <= walk2.index(word2) :
            intersection.append(word2)
        intersections.append(intersection)
        unions.append(unions)

      new_cols = defaultdict(list)
      new_cols[f'w1_index_walk'] = [np.mean([w1.index(clue_node) for w1 in w1_walks_ord if clue_node in w1]) for clue_node in clue_nodes]
      new_cols[f'w2_index_walk'] = [np.mean([w2.index(clue_node) for w2 in w2_walks_ord if clue_node in w2]) for clue_node in clue_nodes]
      new_cols[f'intersection'] = [np.mean([intersect.index(clue_node) for intersect in intersections if clue_node in intersect]) for clue_node in clue_nodes]
      new_cols[f'union'] = [np.mean([union.index(clue_node) for union in unions if clue_node in union]) for clue_node in clue_nodes]
      out.append(pd.concat(
        [group.reset_index(), pd.DataFrame.from_dict(new_cols)],
        axis = 1
      ))

    pd.concat(out).to_csv(
      f'{exp_path}/model_output/indices{"_permuted" if permute else ""}.csv'
    )

if __name__ == "__main__":
  swow = SWOW('../data/exp1')
  np.random.seed(444)
  swow.save_candidates()
  # swow.save_scores('../data/exp1/exp1-cleaned.csv', permute = False)
  # swow.save_scores('../data/exp1/exp1-cleaned.csv', permute = True)
  # swow.save_rank_order('../data/exp1/', permute = False)
  # swow.save_rank_order('../data/exp1/', permute = True)
