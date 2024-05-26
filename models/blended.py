import os
import json
import walker
import warnings

import pandas as pd
import numpy as np
import networkx as nx

from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=FutureWarning)

class blended:
  def __init__(self, exp_path):
    print("Init path:", os.path.abspath('.'))

    # import target words
    self.target_df = pd.read_csv(f"{exp_path}/targets.csv")
    self.target_df["wordpair"]= self.target_df["Word1"]+ "-"+self.target_df["Word2"]
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)
    self.vocab = pd.read_csv(f"{exp_path}/model_input/vocab.csv")
    self.vocab_size = len(list(self.vocab.Word))

    # generated with "python pragmatics.py cdf additive 100 0 0"
    self.sims = pd.read_csv(f"{exp_path}/model_output/speaker_df_allclues.csv")
    with Parallel(n_jobs=2) as parallel:
      parallel(delayed(self.save_candidates)(exp_path, board, word1, word2)
        for word1, word2, board in zip(self.target_df['Word1'], self.target_df['Word2'], self.target_df['boardnames']))

  def get_words_by_node(self, nodes):
    return [self.index_to_name[index] if index in self.index_to_name else None
            for index in nodes]

  def construct_graph(self, exp_path, clues):
    # add bias to association
    associative_strength = pd.read_csv(f'{exp_path}/model_input/swow_strengths.csv').rename(columns={'R123.Strength' : 'weight'})
    clue_bias = self.sims.query(f"targetpair == '{clues[0]}-{clues[1]}'").rename(columns={'clueword' : 'response'})
    blended = pd.merge(associative_strength, clue_bias, how = 'left', on = 'response')
    blended['blended_weight'] = blended['weight'] + 10 * blended['prob']

    # construct graph
    G = nx.from_pandas_edgelist(blended, 'cue', 'response', ['blended_weight'], create_using=nx.DiGraph)
    self.graph = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
    self.index_to_name = {k: v['word'] for k,v in self.graph.nodes(data=True)}
    self.name_to_index = {v['word'] : k for k,v in self.graph.nodes(data=True)}

    # run walks
    start_nodes = [self.name_to_index[name] for name in self.target_words]
    self.rw = walker.random_walks(self.graph, n_walks=1000, walk_len=10000, start_nodes=start_nodes)

  def save_candidates(self, exp_path, board, word1, word2) :
    '''
    write out walks in order of words visited
    '''
    print(word1, word2)
    self.construct_graph(exp_path, board, [word1, word2])
    w1_walks = [x for x in self.rw if x[0] == self.name_to_index[word1]]
    w2_walks = [x for x in self.rw if x[0] == self.name_to_index[word2]]
    d = {f'walk-{int(2*i)}': self.get_words_by_node(w1_walks[i]) for i in range(1000)}
    d.update({f'walk-{int(2*i+1)}': self.get_words_by_node(w2_walks[i]) for i in range(1000)})
    with open(f'{exp_path}/model_output/{word1}-{word2}-walks-blended.json', 'w', encoding ='utf8') as json_file:
      json.dump(d, json_file, ensure_ascii = False)

if __name__ == "__main__":
  np.random.seed(1235)
  swow_exp1 = blended('../data/exp1')
  # swow_exp2 = blended('../data/exp2')
  # swow_exp3 = blended('../data/exp3')
