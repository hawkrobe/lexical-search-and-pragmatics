import random
from random import randrange
import os
import json
import pickle
import itertools
import walker
import math
import warnings
import scipy.spatial.distance
from heapq import nlargest
import pandas as pd
import numpy as np
import networkx as nx
import math
from collections import defaultdict

warnings.simplefilter(action='ignore', category=FutureWarning)

class SWOW:
  '''
  Class description:
    Class model to execute random walks from association norms given by the SWOW model. 
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
    self.rw = walker.random_walks(self.graph, n_walks=n_walks, walk_len=walk_len, start_nodes=indices)

    with open('../data/walk_data/walks.pkl', 'wb') as f:
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
    for path lengths budgeted from 2 - 2^1000. Returns candidate words with likelihoods of being visited. 

    Args:
      w1, w2: target words.
    Returns:
      union_counts, intersection_count : dict of  { budget : {wordid:weight} }    
    '''

    # Retrieve paths that start with target word
    target_indices = self.get_nodes_by_word([w1, w2])
    walks = np.array([x for x in self.rw if x[0] in target_indices]).tolist() 
    
    union_counts = {budget : {i: 0.0001 for i in range(len(self.vocab))} for budget in self.powers_of_two(1000)} 
    intersection_counts = {budget : {i: 0.0001 for i in range(len(self.vocab))} for budget in self.powers_of_two(1000)}

    for search_budget in self.powers_of_two(1000) :  
      for w1_walk, w2_walk in self.chunk(walks, 2) : 
        for element in set(w1_walk[: search_budget]).intersection(w2_walk[: search_budget]) :
          intersection_counts[search_budget][element] += 1
        for element in set(w1_walk[: search_budget]).union(w2_walk[: search_budget]) :
          union_counts[search_budget][element] += 1

    
    # intersection_ids = [
    #   index for (search_budget, d) in intersection_counts.items()
    #   for (index, count) in d.items()
    # ]
    # intersection_words = self.get_words_by_node(intersection_ids)


    # for a given search_budget (e.g. 10 steps)
    # normalize by the total visitation counts and convert index to word
    # TODO Convert the node IDs to strings? self.get_words_by_node([index])[0]

    union_counts_normalized = {budget : {i: 0.0001 for i in range(len(self.vocab))} for budget in self.powers_of_two(1000)} 

    for search_budget, d in union_counts.items():
      for index, count in d.items():
        union_counts_normalized[search_budget][index] = count / sum(d.values()) 

    intersection_counts_normalized = {budget : {i: 0.0001 for i in range(len(self.vocab))} for budget in self.powers_of_two(1000)} 

    for search_budget, d in intersection_counts.items():
      for index, count in d.items():
        intersection_counts_normalized[search_budget][index] = count / sum(d.values()) 

    # # TODO test w/ clue_score
    # for(search_budget, d) in union_counts_normalized.items():
    #     print("budget",  search_budget)
    #     print("d", d) # only storing last value eg word w/ ID 12217 
    #     break

    # intersection_counts_normalized = {
    #   search_budget : { index : count / sum(d.values()) }
    #   for (search_budget, d) in intersection_counts.items()
    #   for (index, count) in d.items()
    # }
    return union_counts_normalized, intersection_counts_normalized

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

    print(clue_indices)

    return ({budget: [d[clue_index] for clue_index in clue_indices] for (budget, d) in union_counts.items()},
            {budget: [d[clue_index] for clue_index in clue_indices] for (budget, d) in intersection_counts.items()})

  def save_candidates(self):
    '''
    Computes union and intersection for all word pairs. Tracks of the number of times a word is visited for different budgets, 
    across all word pairs’ walks. Saves words into union_candidates.json.
    '''
    
    # Loop through word pairs
    unions = {}
    intersections = {}
    for w1, w2 in  zip(self.target_df['Word1'], self.target_df['Word2']) :
      print(w1,w2)
      union_counts, intersection_counts = self.union_intersection_candidates(w1, w2)
      union_candidates = {budget: sorted(d.items(), key=lambda k_v: k_v[1], reverse=True)
                          for (budget, d) in union_counts.items()}
      union_nodes = {budget: [x[0] for x in d] for (budget, d) in union_candidates.items()}
      unions[w1 + '-' + w2] = {'budget=' + str(budget): self.get_words_by_node(d) for (budget, d) in union_nodes.items()}

      intersection_candidates = {budget: sorted(d.items(), key=lambda k_v: k_v[1], reverse=True)
                                 for (budget, d) in intersection_counts.items()}
      intersection_nodes = {budget: [x[0] for x in d] for (budget, d) in intersection_candidates.items()}
      intersections[w1 + '-' + w2] = {'budget=' + str(budget): self.get_words_by_node(d) for (budget, d) in intersection_nodes.items()}

    with open('../data/walk_data/intersection_candidates.json', 'w') as f:
      json.dump(intersections, f)

    with open('../data/walk_data/union_candidates.json', 'w') as f:
      json.dump(unions, f)

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
      union_score, intersect_score = self.clue_score(group['correctedClue'].to_numpy(), 
                                                     group['Word1'].to_numpy()[0], 
                                                     group['Word2'].to_numpy()[0])
      for key in union_score.keys() :
        scores['union' + str(key)].extend(union_score[key])
        scores['intersection' + str(key)].extend(intersect_score[key])
      rows.append(group)

    # save to file
    pd.concat(
      [
        pd.concat(rows,axis=0,ignore_index=True),
        pd.DataFrame.from_dict(scores)
      ],
      axis=1
    ).to_csv('/'.join(data_path.split('/')[:-1])+'/scores.csv')
  
  def midpoint_scores(self, w1, w2):
    # import swow associative embeddings
    embeddings = pd.read_csv("../data/swow_associative_embeddings.csv").transpose().values
    # import vocab
    w1_vec = embeddings[self.vocab.index(w1)]
    w2_vec = embeddings[self.vocab.index(w2)]
    midpoint = (w1_vec + w2_vec)/2
    midpoint = midpoint.reshape((1, embeddings.shape[1]))
    similarities = 1 - scipy.spatial.distance.cdist(midpoint, embeddings, 'cosine')
    y = np.array(similarities)
    y_sorted = np.argsort(-y).flatten() ## gives sorted indices
    closest_words = [self.vocab[i] for i in y_sorted]
    return closest_words

  def save_midpoint_scores(self, data_path):
    expdata = pd.read_csv(f"{data_path}", encoding= 'unicode_escape')
    scores = defaultdict(list)
    rows = []
    for name, group in expdata.groupby('wordpair') :
      closest_to_midpoint = self.midpoint_scores(group['Word1'].to_numpy()[0], group['Word2'].to_numpy()[0])
      for search_budget in self.powers_of_two(1000) :
        print("search_budget=",search_budget)
        clue_list = group['correctedClue'].to_numpy()
        for clue in clue_list:
          if clue in closest_to_midpoint[:search_budget]:
            prop = 1- closest_to_midpoint[:search_budget].index(clue) / len(closest_to_midpoint[:search_budget])
            scores['mid' + str(search_budget)].append(prop)
          else:
            scores['mid' + str(search_budget)].append(0)
      rows.append(group)
    
    pd.concat(
      [
        pd.concat(rows,axis=0,ignore_index=True),
        pd.DataFrame.from_dict(scores)
      ],
      axis=1
    ).to_csv('/'.join(data_path.split('/')[:-1])+'/midpoint_scores.csv')

  def save_frequency_scores(self,data_path):
    expdata = pd.read_csv(f"{data_path}", encoding= 'unicode_escape')
    scores = defaultdict(list)
    rows = []
    freq = list(pd.read_csv("../data/vocab.csv").sort_values(by="LgSUBTLWF", ascending=False)["Word"])
    print(freq)
    # sort by frequency 
    for name, group in expdata.groupby('wordpair') :
      for search_budget in self.powers_of_two(1000) :
        print("search_budget=",search_budget)
        clue_list = group['correctedClue'].to_numpy()
        for clue in clue_list:
          if clue in freq[:search_budget]:
            prop = 1 - freq[:search_budget].index(clue) / len(freq[:search_budget])
            scores['freq' + str(search_budget)].append(prop)
          else:
            scores['freq' + str(search_budget)].append(0)
      rows.append(group)
    
    pd.concat(
      [
        pd.concat(rows,axis=0,ignore_index=True),
        pd.DataFrame.from_dict(scores)
      ],
      axis=1
    ).to_csv('/'.join(data_path.split('/')[:-1])+'/freq_scores.csv')
  
  def get_example_walk(self, w1, w2):
    target_indices = self.get_nodes_by_word([w1, w2])
    walks = np.array([x for x in self.rw if x[0] in target_indices]).tolist()
    random_index =  randrange(0, 999)
    w1_walk = self.get_words_by_node(walks[random_index])
    w2_walk = self.get_words_by_node(walks[random_index+1])

    intersection_counts = {budget : defaultdict(lambda: 0.000001) for budget in self.powers_of_two(1000)}
    union_counts = {budget : defaultdict(lambda: 0.000001) for budget in self.powers_of_two(1000)}

    for search_budget in self.powers_of_two(1000) :
        intersection = list(set(w1_walk[: search_budget]).intersection(w2_walk[: search_budget]))
        intersection_counts[search_budget] = intersection
        union = list(set(w1_walk[: search_budget]).union(w2_walk[: search_budget]))
        union_counts[search_budget] = union

    with open('../data/walk_data/example_intersection.json', 'w') as f:
      json.dump(intersection_counts, f)

    with open('../data/walk_data/example_union.json', 'w') as f:
      json.dump(union_counts, f)

    with open('../data/walk_data/example_walk.json', 'w') as f:
      json.dump({w1_walk[0]:w1_walk, w2_walk[0]: w2_walk}, f)
  
  def visualize(self, w1, w2):
    self.get_example_walk(w1,w2)
    with open('../data/walk_data/example_walk.json') as json_file:
      walks = json.load(json_file)
    with open('../data/walk_data/example_union.json') as json_file:
      union = json.load(json_file)
    with open('../data/walk_data/example_intersection.json') as json_file:
      intersection = json.load(json_file)
    
    labels = nx.get_node_attributes(self.graph, 'word')
  
    sub = list(set(walks[w1] + walks[w2]))
    sub_indices = self.get_nodes_by_word(sub)

    X = self.graph.subgraph(sub_indices)
    
    X = nx.relabel_nodes(X, labels)
    
    

    it1 = walks[w1][:17]
    w1_walk_edges = list(zip(it1, it1[1:]))
    Y = X.subgraph(it1)
    Y_edges = list(Y.edges())
    Z1 = nx.Graph(Y)
    Z1.remove_edges_from(Y_edges)
    Z1.add_edges_from(w1_walk_edges)

    it2 = walks[w2][:17]
    w2_walk_edges = list(zip(it2, it2[1:]))
    Y = X.subgraph(it2)
    Y_edges = list(Y.edges())
    Z2 = nx.Graph(Y)
    Z2.remove_edges_from(Y_edges)
    Z2.add_edges_from(w2_walk_edges)

    intX = intersection['256']
    int_graph =X.subgraph(intX)
    int_graph2 = nx.Graph(int_graph)
    int_graph2.remove_edges_from(list(Z1.edges()))
    int_graph2.remove_edges_from(list(Z2.edges()))

    unionX =union['256']
    union_graph =X.subgraph(unionX)
    union_graph2 = nx.Graph(union_graph)
    union_graph2.remove_edges_from(list(Z1.edges()))
    int_graph2.remove_edges_from(list(Z2.edges()))

    pos = nx.spring_layout(X, k=5/math.sqrt(X.order()))
    # draw main graph
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)

    nx.draw_networkx(X.subgraph(walks[w1]), pos=pos, font_size=1, node_color='#DDE6FF', font_color='0.9', node_size=0.1,edge_color = '#DDE6FF')
    nx.draw_networkx(X.subgraph(walks[w2]), pos=pos, font_size=1, node_color='#EEFBE8', font_color='0.9', node_size=0.1,edge_color = '#EEFBE8')

    nx.draw_networkx(union_graph2, pos=pos, font_size=12, node_color='#FADD9E', font_color='black', node_size = 5, edge_color = '#FADD9E')
    nx.draw_networkx(int_graph2, pos=pos, font_size=12, node_color='#FADD9E', font_color='black', node_size = 5, edge_color = '#FADD9E')
    

    nx.draw_networkx(Z1, pos=pos, font_size=12, node_color='red', font_color='red', edge_color = "red" , node_size = 5)
    nx.draw_networkx(Z2, pos=pos, font_size=12, node_color='blue', font_color='blue', edge_color = "blue" , node_size = 5)

    plt.tight_layout()
    plt.savefig("visualize_graph.png", format="PNG")
    #plt.show()


    
if __name__ == "__main__":
  # current dir is models
  # swow = SWOW('../data') 

  # Chang pathed for debugging
  os.chdir("./models")
  print("Main method path: ", os.path.abspath('.'))

  swow = SWOW('../data') 
  np.random.seed(44)

  # swow.union_intersection_candidates('cave', 'knight')
  unions, intersections = swow.union_intersection_candidates('cave', 'knight') # TODO ur inputs
  # print(unions)
  # print(intersections)
  print(swow.clue_score(["animal"], "lion", "tiger")) 
  # swow.save_scores('../data/TEST/e1_data_long.csv') # TODO eg e1 contains OOVs. search for OOV words? Run whole file, wil have Nones
  # TODO Nones when outputing scores files 


  # swow.save_scores('../data/exp2/e2_corrected.csv')
  # swow.save_scores('../data/exp1/e1_data_long.csv')
  #swow.save_candidates()
  #swow.midpoint_scores('happy','sad')
  #swow.save_midpoint_scores('../data/exp1/e1_data_long.csv')
  #swow.save_frequency_scores('../data/exp1/e1_data_long.csv')
  #swow.get_example_walk("happy", "sad")
  #swow.visualize('cave', 'knight')
