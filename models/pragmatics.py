import json
import sys
import pickle
import itertools
import warnings
import scipy

import scipy.spatial.distance

import pandas as pd
import numpy as np
import networkx as nx

warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.special import softmax, expit
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

class Selector:
  def __init__(self, exp_path, params) :
    # handle parameters
    self.cost_type = params[0] if len(params) > 0 else 'cdf'
    self.inf_type = params[1] if len(params) > 1 else 'additive'
    self.alpha = float(params[2]) if len(params) > 2 else None
    self.costweight = float(params[3]) if len(params) > 3 else None
    self.distweight = float(params[4]) if len(params) > 4 else None
    if self.inf_type == 'additive' :
      assert(self.distweight is not None)
    else :
      assert(self.alpha is not None and self.costweight is not None)

    # read in metadata
    self.exp_path = exp_path
    self.vocab = pd.read_csv(f"{exp_path}/model_input/vocab.csv")
    self.targets = pd.read_csv(f"{exp_path}/targets.csv")
    self.cluedata = pd.read_csv(f"{exp_path}/cleaned.csv")
    self.embeddings = (
      pd.read_csv(f"{exp_path}/model_input/swow_embeddings.csv").transpose().values
    )
    with open(f'{exp_path}/boards.json', 'r') as json_file:
      self.boards = json.load(json_file)

    # initialize/cache objects
    self.create_board_combos()
    self.create_cost_fn()
    self.sims = {
      boardname : self.create_sim_matrix(boardname)
      for boardname in self.boards.keys()
    }

  def create_cost_fn(self) :
    print('building cost fn...')

    # transform measures to costs
    measure_df = pd.read_csv(f"{self.exp_path}/model_output/{self.cost_type}s_long.csv")
    if self.cost_type == 'freq':
      measure_df.loc[:,'value'] = -1 * measure_df.loc[:,'value']
    else :
      measure_df.loc[:,'value'] = -1 * np.log(0.01 + measure_df.loc[:,'value'])

    self.cost = {}
    for measure in measure_df['measure'].unique() :
      subset = measure_df.query("measure == @measure")
      self.cost[measure] = {
        wordpair: subset.query("wordpair == @wordpair")['value'].to_numpy()
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

  def informativity(self, boardname, targetpair_idx) :
    if self.inf_type == 'RSA' :
      diagnosticity = np.log(self.literal_guesser(boardname))[targetpair_idx].ravel()
    elif self.inf_type == 'additive' :
      distractors = np.delete(self.sims[boardname], targetpair_idx, axis=0)
      assert(distractors.shape == (189, 12218))
      diagnosticity = -np.max(distractors, axis=0)
    elif self.inf_type == 'no_prag' :
      diagnosticity = 0
    return ((1-self.distweight) * self.sims[boardname][targetpair_idx].ravel() 
             + self.distweight * diagnosticity)

  def literal_guesser(self, boardname):
    '''
    literal guesser probability over each wordpair
    '''
    return softmax(self.sims[boardname], axis = 0) # 190 x vocab

  def pragmatic_speaker(self, targetpair, boardname, cost_fn, listofclues= None):
    '''
    softmax likelihood of each possible clue
    '''

    targetpair_idx = list(self.board_combos[boardname]['wordpair']).index(targetpair)
    inf = self.informativity(boardname, targetpair_idx)
    cost = self.cost[cost_fn][targetpair].ravel()
    utility = (1-self.costweight) * inf - self.costweight * cost
    if(listofclues is not None):
        # find indices of listofclues in the vocab
        listofclues_idx = [list(self.vocab["Word"]).index(clue) for clue in listofclues if clue in list(self.vocab["Word"])]
        utility = utility[listofclues_idx]
    return softmax(self.alpha * utility)
  
  def get_speaker_scores(self, boarddata, probsarray, listofclues = None) :
    '''
    takes a set of clues and word pairs, and computes the probability and rank of each clue
    inputs:
    (1) boarddata
    (2) probsarray: a candidates array of pragmatic speaker predictions

    outputs:
    softmax probabilities and ranks for each candidate in cluedata
    '''
    speaker_probs = []
    for index, row in boarddata.iterrows():
        if row["correctedClue"] in list(self.vocab["Word"]):
            if(listofclues is not None):
              clue_index = listofclues.index(row["correctedClue"])
            else:
              clue_index = list(self.vocab["Word"]).index(row["correctedClue"])
            if clue_index < len(probsarray):
              speaker_probs.append(probsarray[clue_index])
            else:
              speaker_probs.append("NA")
        else:
            speaker_probs.append("NA")
    return speaker_probs

  def get_speaker_df(self):
    '''
    returns a complete dataframe of pragmatic speaker ranks & probabilities over different representations
    over a given set of candidates

    inputs:
    params: optimal parameter dictionary for different representation keys

    outputs:
    a dataframe with clue ranks & probs for each possible candidate & representation
    '''
    boards = []
    for cost_fn in self.cost.keys() :
      for index, row in self.targets.iterrows() :
        boardname = row["boardnames"]
        targetpair = row['wordpair']
        y = self.pragmatic_speaker(targetpair, boardname, cost_fn)
        #y = self.pragmatic_speaker(targetpair, boardname, cost_fn,clues)
        expdata_board = self.cluedata.copy().query("wordpair == @targetpair and boardnames == @boardname")
        speaker_prob = self.get_speaker_scores(expdata_board, y)
        #speaker_prob = self.get_speaker_scores(expdata_board, y, clues)
        expdata_board.loc[:,"cost_fn"] = cost_fn
        expdata_board.loc[:,"alpha"] = self.alpha
        expdata_board.loc[:,"costweight"] = self.costweight
        expdata_board.loc[:,"distweight"] = self.distweight        
        expdata_board.loc[:,"model"] = self.inf_type + self.cost_type
        expdata_board.loc[:,"prob"] = speaker_prob
        boards.append(expdata_board)
    return pd.concat(boards)

  def get_likelihood(self, params) :
    softplus = lambda x: np.log1p(np.exp(x))
    self.alpha = softplus(params[0])
    self.costweight = expit(params[1])
    self.distweight = 0# expit(params[2])
    df = self.get_speaker_df()
    lik = -np.sum(np.log(np.asarray(df['prob'])[~np.isnan(df['prob'])]))
    print(self.alpha, self.costweight, '(', self.distweight, ')', ':', lik)
    return lik

  def optimize(self) :
    return scipy.optimize.minimize(self.get_likelihood, [8, 0.1]) 
  
if __name__ == "__main__":
  # cdf / freq
  exp_path = '../data/exp1/'
  selector = Selector(exp_path, sys.argv[1:])
  # selector.optimize()
  
  out = selector.get_speaker_df()
  out.to_csv(
    f'{exp_path}/model_output/speaker_df_{selector.cost_type}_{selector.inf_type}_{sys.argv[6]}.csv'
  )
