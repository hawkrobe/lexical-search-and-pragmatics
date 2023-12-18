import json
import sys
import pickle
import itertools
import warnings
import scipy.spatial.distance

import pandas as pd
import numpy as np
import networkx as nx

warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.special import softmax
from joblib import Parallel, delayed
from tqdm import tqdm

class RSA:
  def __init__(self, exp_path, inf_type, cost_type) :
    self.exp_path = exp_path
    self.inf_type = inf_type
    self.cost_type = cost_type
    self.vocab = pd.read_csv(f"{exp_path}/model_input/vocab.csv")
    self.targets = pd.read_csv(f"{exp_path}/targets.csv")
    self.cluedata = pd.read_csv(f"{exp_path}/cleaned.csv")
    self.embeddings = (
      pd.read_csv(f"{exp_path}/model_input/swow_embeddings.csv").transpose().values
    )
    with open(f'{exp_path}/boards.json', 'r') as json_file:
      self.boards = json.load(json_file)

    self.create_board_combos()
    self.create_cost_fn()
    self.sims = {
      boardname : self.create_sim_matrix(boardname)
      for boardname in self.boards.keys()
    }

  def create_cost_fn(self) :
    print('building cost fn...')
    self.cost = {}
    measure_df = pd.read_csv(f"{self.exp_path}/model_output/{self.cost_type}s_long.csv")

    # transform measures to costs -- higher rank means more costly but
    # lower score is more costly (need to invert)
    if self.cost_type == 'rank' :
      measure_df.loc[:,'value'] = np.log1p(measure_df.loc[:,'value'])
    elif self.cost_type == 'freq':
      measure_df.loc[:,'value'] = -1 * measure_df.loc[:,'value']
    else :
      measure_df.loc[:,'value'] = -1 * np.log(0.001 + measure_df.loc[:,'value'])

    for measure in measure_df['measure'].unique() :
      subset = measure_df.query("measure == @measure")
      self.cost[measure] = {
        wordpair: subset.query("wordpair == @wordpair")
        for wordpair in self.cluedata['wordpair'].unique()
      }
    
  def create_board_combos(self):
    '''
    calculates all pairwise combinations of the words on the board
    '''
    self.board_combos = {}
    for board_name, board in self.boards.items() :
      all_possible_combs = list(itertools.combinations(board, 2))
      combo_df = pd.DataFrame(all_possible_combs, columns =['Word1', 'Word2'])
      combo_df["wordpair"] = combo_df["Word1"] + '-'+ combo_df["Word2"]
      self.board_combos[board_name] = combo_df

  def create_sim_matrix(self, boardname):
    '''
    product similarities of given vocab to each wordpair
    '''

    combo_df = self.board_combos[boardname]
    context_board = self.boards[boardname]

    # grab subset of words in given board and their corresponding semantic reps
    board_df = self.vocab.query('Word in @context_board')
    board_vectors = self.embeddings[list(board_df.index)]

    ## clue_sims is the similarity of ALL clues in full candidate space to EACH word on board (size 20)
    clue_sims = 1 - scipy.spatial.distance.cdist(board_vectors, self.embeddings, 'cosine')

    ## next we take thesimilarities between c-w1 and c-w2 for that
    ## specific board's 190 word-pairs.
    board_df.reset_index(inplace = True)
    f_w1 = [clue_sims[board_df[board_df["Word"]==row["Word1"]].index.values[0]]
            for  index, row in combo_df.iterrows()]
    f_w2 = [clue_sims[board_df[board_df["Word"]==row["Word2"]].index.values[0]]
            for  index, row in combo_df.iterrows()]

    # note that cosine is in range [-1, 1] so we have to convert to [0,1] for
    # the product semantics to be valid
    return ((np.array(f_w1) + 1) /2) * ((np.array(f_w2) + 1)/2)

  def literal_guesser(self, boardname):
    '''
    literal guesser probability over each wordpair
    '''
    return softmax(self.sims[boardname], axis = 0) # 190 x vocab

  def pragmatic_speaker(self, targetpair, boardname, modelname, alpha, costweight):
    '''
    softmax likelihood of each possible clue
    '''

    targetpair_idx = list(self.board_combos[boardname]['wordpair']).index(targetpair)
    inf = (
      np.log(self.literal_guesser(boardname))[targetpair_idx].ravel()
      if self.inf_type == 'prag'
      else np.log(self.sims[boardname])[targetpair_idx].ravel()
    )
    cost = self.cost[modelname][targetpair]['value'].to_numpy().ravel()
    utility = (1-costweight) * inf - costweight * cost
    return softmax(alpha * utility)

  def get_speaker_scores(self, boarddata, probsarray, probsarray_sorted) :
    '''
    takes a set of clues and word pairs, and computes the probability and rank of each clue
    inputs:
    (1) boarddata
    (2) probsarray: a candidates array of pragmatic speaker predictions
    (3) probsarray_sorted: a sorted candidates array of pragmatic speaker predictions

    outputs:
    softmax probabilities and ranks for each candidate in cluedata
    '''
    speaker_probs = []
    speaker_ranks = []
    for index, row in boarddata.iterrows():
        if row["correctedClue"] in list(self.vocab["Word"]):
            clue_index = list(self.vocab["Word"]).index(row["correctedClue"])
            speaker_probs.append(probsarray[clue_index])
            speaker_ranks.append(np.nonzero(probsarray_sorted==clue_index)[0][0])
        else:
            speaker_probs.append("NA")
            speaker_ranks.append("NA")
    return speaker_probs, speaker_ranks

  def get_speaker_df(self, params):
    '''
    returns a complete dataframe of pragmatic speaker ranks & probabilities over different representations
    over a given set of candidates

    inputs:
    params: optimal parameter dictionary for different representation keys

    outputs:
    a dataframe with clue ranks & probs for each possible candidate & representation
    '''
    speakerprobs_dfs = []
    for index, row in self.targets.iterrows():
      boardname = row["boardnames"]
      targetpair = row['wordpair']
      y = self.pragmatic_speaker(targetpair, boardname, params[0], params[1], params[2])
      y_sorted = np.argsort(-y)
      expdata_board = self.cluedata.query("wordpair == @targetpair and boardnames == @boardname").copy()
      speaker_prob, speaker_rank = self.get_speaker_scores(expdata_board, y, y_sorted)
      expdata_board.loc[:,"cost"] = params[0]
      expdata_board.loc[:,"alpha"] = params[1]
      expdata_board.loc[:,"costweight"] = params[2]
      expdata_board.loc[:,"model"] = self.inf_type + self.cost_type
      expdata_board.loc[:,"prag_speaker_probs"] = speaker_prob
      expdata_board.loc[:,"prag_speaker_rank"] = speaker_rank
      speakerprobs_dfs.append(expdata_board)

    return pd.concat(speakerprobs_dfs)

if __name__ == "__main__":
  # cdf / freq
  cost_type = sys.argv[1] if len(sys.argv) > 1 else 'cdf'
  inf_type = sys.argv[2] if len(sys.argv) > 2 else 'prag'
  exp_path = '../data/exp2/'
  rsa = RSA(exp_path, inf_type, cost_type)
  param_grid = itertools.product(
    rsa.cost.keys(),
    [1, 2, 4, 8, 16, 32, 64],
    [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]
  )
  out = pd.concat(list(tqdm(
    Parallel(n_jobs=8, return_as = 'generator')(
      delayed(rsa.get_speaker_df)(params) for params in param_grid
    ),
    total = len(rsa.cost.keys()) * 6 * 9
  )))
  out.to_csv(
    f'{exp_path}/model_output/speaker_df_{cost_type}_{inf_type}.csv'
  )
