import random
import os
import json
import pickle
import itertools
import walker
import math
import warnings
import scipy.spatial.distance

import pandas as pd
import numpy as np
import networkx as nx

from collections import defaultdict

warnings.simplefilter(action='ignore', category=FutureWarning)

class SWOW:
  def __init__(self, data_path):
    self.load_graph(data_path)
    self.load_random_walks(data_path)

    # Import boards
    with open(f'{data_path}/boards.json', 'r') as json_file:
        self.boards = json.load(json_file)

    # import empirical clues (cleaned)
    self.expdata = pd.read_csv(f"{data_path}/final_board_clues_all.csv", 
                               encoding= 'unicode_escape')

    # import target words
    self.target_df = pd.read_csv(f"{data_path}/connector_wordpairs_boards.csv".format())
    self.target_df["wordpair"]= self.target_df["Word1"]+ "-"+self.target_df["Word2"]
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)

    # import embeddings
    # self.embeddings = pd.read_csv(f"{data_path}/swow_associative_embeddings.csv").transpose().values
    # self.embeddings_vocab = pd.read_csv(f"{data_path}/swow_associative_embeddings.csv").columns

  def load_graph(self, data_path):
    '''
    reads in networkx graph saved as pickle
    '''
    if os.path.exists(f'{data_path}/swow.gpickle') :
      with open(f'{data_path}/swow.gpickle', 'rb') as f:
        self.graph = pickle.load(f)
    else :
      self.save_graph(data_path, None)

  def save_graph(self, path, threshold):
    '''
    creates graph directly from pandas edge list and saves to file
    '''
    edges = pd.read_csv(path).rename(columns={'R123.Strength' : 'weight'})
    G = nx.from_pandas_edgelist(edges, 'cue', 'response', ['weight'], create_using=nx.DiGraph)
    G = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
    with open('../../data/swow.gpickle', 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

  def load_random_walks(self, data_path):
    '''
    runs n_walks independent random walks of walk_len length from each words
    '''
    if os.path.exists(f'{data_path}/walks.pkl'):
      with open(f'{data_path}/walks.pkl', 'rb') as f:
        self.rw = pickle.load(f)
    else :
      self.save_random_walks()

  def save_random_walks(self, n_walks = 1000, walk_len = 10000):
    '''
    runs n_walks independent random walks of walk_len length from each words
    '''
    indices = self.get_nodes_by_word(self.target_words)
    self.rw = walker.random_walks(self.graph, n_walks=n_walks, walk_len=walk_len, start_nodes=indices)
    with open('../../data/walks.pkl', 'wb') as f:
      pickle.dump(self.rw, f)


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
    return [k for k, v in self.graph.nodes(data=True)
              for word in words
              if v['word'] == word]

  def get_words_by_node(self, nodes):
    '''
    looks up which words a list of node IDs correspond to
    '''
    return [self.graph.nodes[node]['word'] for node in nodes]

  def union_intersection_candidates(self, w1, w2):
    '''
    return a list of candidates sorted by likelihood of being visited
    '''

    # looks up row for this target pair
    target_indices = self.get_nodes_by_word([w1, w2])
    walks = np.array([x for x in self.rw if x[0] in target_indices]).tolist()
    union_counts = {budget : defaultdict(lambda: 0.001) for budget in self.powers_of_two(1000)}
    intersection_counts = {budget : defaultdict(lambda: 0.001) for budget in self.powers_of_two(1000)}
    for search_budget in self.powers_of_two(1000) :
      for w1_walk, w2_walk in self.chunk(walks, 2) :
        for element in set(w1_walk[: search_budget]).intersection(w2_walk[: search_budget]) :
          intersection_counts[search_budget][element] += 1
        for element in set(w1_walk[: search_budget]).union(w2_walk[: search_budget]) :
          union_counts[search_budget][element] += 1

    return union_counts, intersection_counts

  def clue_score(self, clues, w1, w2):
    clue_indices = self.get_nodes_by_word(clues)
    union_counts, intersection_counts = self.union_intersection_candidates(w1, w2)
    return ({budget: [d[clue_index] for clue_index in clue_indices] for (budget, d) in union_counts.items()},
            {budget: [d[clue_index] for clue_index in clue_indices] for (budget, d) in intersection_counts.items()})

  def save_candidates(self):
    # Loop through word pairs
    unions = {}
    intersections = {}
    for w1, w2 in  zip(swow.target_df['Word1'], swow.target_df['Word2']) :
      print(w1,w2)
      union_counts, intersection_counts = swow.union_intersection_candidates(w1, w2)
      union_candidates = {budget: sorted(d.items(), key=lambda k_v: k_v[1], reverse=True)
                          for (budget, d) in union_counts.items()}
      union_nodes = {budget: [x[0] for x in d] for (budget, d) in union_candidates.items()}
      unions[w1 + '-' + w2] = {'budget=' + str(budget): swow.get_words_by_node(d) for (budget, d) in union_nodes.items()}

      intersection_candidates = {budget: sorted(d.items(), key=lambda k_v: k_v[1], reverse=True)
                                 for (budget, d) in intersection_counts.items()}
      intersection_nodes = {budget: [x[0] for x in d] for (budget, d) in intersection_candidates.items()}
      intersections[w1 + '-' + w2] = {'budget=' + str(budget): swow.get_words_by_node(d) for (budget, d) in intersection_nodes.items()}

    with open('intersection_candidates.json', 'w') as f:
      json.dump(intersections, f)

    with open('union_candidates.json', 'w') as f:
      json.dump(unions, f)

  def save_scores(self):
    scores = defaultdict(list)

    # look up how often each clue was visited
    for name, group in swow.expdata.groupby('wordpair') :
      union_score, intersect_score = swow.clue_score(group['Clue1'].to_numpy(), group['Word1'].to_numpy()[0], group['Word2'].to_numpy()[0])
      for key in union_score.keys() :
        scores['union' + str(key)].extend(union_score[key])
        scores['intersection' + str(key)].extend(intersect_score[key])

    # save to file
    pd.concat(
      [swow.expdata.copy(), pd.DataFrame.from_dict(scores)],
      axis=1
    ).to_csv(
      '../../data/scores.csv'
    )


if __name__ == "__main__":
  swow = SWOW('../../data')
  np.random.seed(44)
  swow.save_scores()
