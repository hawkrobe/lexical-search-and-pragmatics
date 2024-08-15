import json
import sys
import itertools
import warnings
import scipy
from functools import lru_cache

import scipy.spatial.distance

import pandas as pd
import numpy as np
import networkx as nx

warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.special import softmax

class Selector:
  def __init__(self, exp_path, cost_type = 'none', inf_type = 'RSA') :
    # handle parameters
    self.cost_type = cost_type
    self.inf_type = inf_type

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
    self.cost = {}
    if self.cost_type == 'none':
      return 
    
    # transform measures to costs
    measure_df = pd.read_csv(f"{self.exp_path}/model_output/{self.cost_type}s_long.csv")
    measure_df.loc[:,'value'] = -1 * measure_df.loc[:,'value']
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

  @lru_cache(maxsize=None)
  def literal_guesser(self, boardname):
    '''
    literal guesser probability over each wordpair
    '''
    return softmax(50*self.sims[boardname], axis = 0)

  @lru_cache(maxsize=None)
  def diagnosticity(self, boardname, targetpair) :
    targetpair_idx = list(self.board_combos[boardname]['wordpair']).index(targetpair)
    if self.inf_type == 'RSA' :
      return self.literal_guesser(boardname)[targetpair_idx].ravel()
    elif self.inf_type == 'additive' :
      distractors = np.delete(self.sims[boardname], targetpair_idx, axis=0)
      assert(distractors.shape == (189, 12218))
      return -np.max(distractors, axis=0)
    elif self.inf_type == 'no_prag' :
      return 0

  def fit(self, boardname, targetpair) :
    targetpair_idx = list(self.board_combos[boardname]['wordpair']).index(targetpair)
    return self.sims[boardname][targetpair_idx].ravel()

  def informativity(self, distweight, boardname, targetpair) :
    return ((1-distweight) * self.fit(boardname, targetpair)
             + distweight * self.diagnosticity(boardname, targetpair))

  def pragmatic_speaker(self, targetpair, boardname, cost_fn, clueset):
    '''
    softmax likelihood of each possible clue
    '''
    inf = self.informativity(self.distweight, boardname, targetpair)
    cost = self.cost[cost_fn][targetpair].ravel() if self.cost_type != 'none' else 0
    utility = (1-self.costweight) * inf - self.costweight * cost
    return softmax(self.alpha * utility[clueset])

  def get_speaker_df(self, clues_only = False):
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
        boarddata = self.cluedata.copy().query("wordpair == @targetpair and boardnames == @boardname")
        vocab = list(self.vocab["Word"])
        clue_indices = [vocab.index(word) for word in boarddata['correctedClue'] if word in vocab] if clues_only else range(len(vocab))
        clueset = self.vocab["Word"][clue_indices]
        y = self.pragmatic_speaker(targetpair, boardname, cost_fn, clue_indices)
        speaker_probs = [
          y[list(clueset).index(row["correctedClue"])]
          if row["correctedClue"] in vocab else 'NA'
          for i, row in boarddata.iterrows()
        ]
        boarddata.loc[:,"cost_fn"] = cost_fn
        boarddata.loc[:,"alpha"] = self.alpha
        boarddata.loc[:,"costweight"] = self.costweight
        boarddata.loc[:,"distweight"] = self.distweight
        boarddata.loc[:,"model"] = self.inf_type + self.cost_type
        boarddata.loc[:,"prob"] = speaker_probs
        boards.append(boarddata)
    return pd.concat(boards)

  def get_spearman(self, params) :
    softplus = lambda x: np.log1p(np.exp(x))
    self.alpha = 1         # alpha is irrelevant because spearman only looks at ranks
    self.costweight = 0    # fix costweight at 0 for the purposes of exp 3 comparison
    self.distweight = params[0]
    df = self.get_speaker_df(clues_only=False)
    combo = self.human_df.merge(df, on=['wordpair', 'correctedClue'])
    combo.loc[:, 'prob_numeric'] = pd.to_numeric(combo['prob'], errors = 'coerce')
    corr = (combo.groupby(['cost_fn']).apply(
      lambda d: d['prob_numeric'].corr(d['response'], method='spearman')
    ).max())
    print(self.alpha, self.costweight, '(', self.distweight, ')', ':', corr)
    return -corr

  def optimize(self, fn) :
    self.human_df = pd.read_csv(f"{self.exp_path}/model_input/human-ratings.csv")
    return scipy.optimize.brute(selector.get_spearman, (slice(0,1.1,.1),))

  def print_examples(self) :
    target_idx = list(selector.board_combos['board15']['wordpair']).index('lion-tiger')
    fit = selector.fit('board15', target_idx)
    diag = selector.diagnosticity('board15', target_idx)
    cost = selector.cost[128]['lion-tiger']
    utility = selector.pragmatic_speaker('lion-tiger', 'board15', 256, range(len(selector.vocab)))
    print('word, fit, diagnositicity, cost, speaker prob')
    for word in ['cat', 'animal', 'claws', 'hiss'] :
      idx = list(selector.vocab["Word"]).index(word)
      print(word, ':', fit[idx], diag[idx], cost[idx], utility[idx])
      distractor_pairs = np.delete(list(self.board_combos['board15']['wordpair']), target_idx, axis=0)
      distractors = np.delete(self.sims['board15'], target_idx, axis=0)
      print('biggest distractor', distractor_pairs[np.argmax(distractors,axis=0)[idx]])

if __name__ == "__main__":
  exp_path = '../data/exp1/'
  selector = Selector(exp_path, sys.argv[1:])
  #selector.print_examples()
  # out = selector.get_speaker_df()
  # out.to_csv(
  #   f'{exp_path}/model_output/speaker_df_{selector.cost_type}_{selector.inf_type}_{sys.argv[6]}.csv'
  # )

  out = selector.get_all_clues()
  out.to_csv(
    f'{exp_path}/model_output/speaker_df_allclues.csv'
  )

  # exp_path = '../data/exp3/'
  # selector = Selector(exp_path, sys.argv[1:])

  # find optimal parameters
  # out = selector.get_speaker_df()
  # out.to_csv(
  #   f'{exp_path}/model_output/speaker_df_{selector.cost_type}_{selector.inf_type}_{sys.argv[6]}.csv'
  # )
