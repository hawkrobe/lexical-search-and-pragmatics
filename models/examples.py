import json
import sys
import pickle
import itertools
import warnings
import scipy
from functools import lru_cache

import scipy.spatial.distance

import pandas as pd
import numpy as np
import networkx as nx
from tabulate import tabulate


warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.special import softmax, expit, logit
from scipy.stats import norm
#from joblib import Parallel, delayed, parallel_config
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
    measure_df.loc[:,'value'] = -1 * measure_df.loc[:,'value']

    self.cost = {}
    for measure in measure_df['measure'].unique() :
      subset = measure_df.query("measure == @measure")
      self.cost[measure] = {
        wordpair: subset.query("wordpair == @wordpair")['value'].to_numpy()
        for wordpair in self.cluedata['wordpair'].unique()
      }
      # at this point, we want to check which dict values are empty and use the flipped version

      # find dict values that are empty
      empty = [k for k, v in self.cost[measure].items() if len(v) == 0]

      # for each empty value, flip the wordpair and find the corresponding value inside subset

      for wordpair in empty :
        flipped = wordpair.split('-')[1] + '-' + wordpair.split('-')[0]
        self.cost[measure][wordpair] = subset.query("wordpair == @flipped")['value'].to_numpy()

    
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
    #print("context_board", context_board)

    # grab subset of words in given board and their corresponding semantic reps
    board_df = self.vocab.query('Word in @context_board')
    #print("board_df", board_df)
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
    x = softmax(50*self.sims[boardname], axis = 0)
    print(f"shape of x: {x.shape}")
    return softmax(50*self.sims[boardname], axis = 0)

  @lru_cache(maxsize=None)
  def diagnosticity(self, boardname, targetpair_idx) :
    if self.inf_type == 'RSA' :
      return self.literal_guesser(boardname)[targetpair_idx].ravel()
    elif self.inf_type == 'additive' :
      distractors = np.delete(self.sims[boardname], targetpair_idx, axis=0)
      #assert(distractors.shape == (189, 12218))
      return -np.max(distractors, axis=0) if len(distractors) > 0 else np.zeros(12218)
    elif self.inf_type == 'no_prag' :
      return 0

  def fit(self, boardname, targetpair_idx) :
    return self.sims[boardname][targetpair_idx].ravel()

  def informativity(self, boardname, targetpair_idx) :
    return ((1-self.distweight) * self.fit(boardname, targetpair_idx)
             + self.distweight * self.diagnosticity(boardname, targetpair_idx))

  def pragmatic_speaker(self, targetpair, boardname, cost_fn, clueset):
    '''
    softmax likelihood of each possible clue
    '''
    #print("targetpair=", targetpair)
    flipped = 0 if targetpair in list(self.board_combos[boardname]['wordpair']) else 1
    flipped_targetpair = targetpair.split('-')[1] + '-' + targetpair.split('-')[0]
    targetpair_idx = list(self.board_combos[boardname]['wordpair']).index(targetpair) if not flipped else list(self.board_combos[boardname]['wordpair']).index(flipped_targetpair)
    inf = self.informativity(boardname, targetpair_idx)
    
    flipped = 0 if targetpair in list(self.cost[cost_fn].keys()) else 1
  
    cost = self.cost[cost_fn][targetpair].ravel() if not flipped else self.cost[cost_fn][flipped_targetpair].ravel()
    
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

  def print_examples(self, targets, cost_fn_fit, candidates) :
    # find the board with the target wordpair
    boardname = self.targets.query("wordpair == @targets")['boardnames'].values[0]
    print('board:', boardname)
    # target flip order
    #flipped_targets = '-'.join(targets.split('-')[::-1])
    #unflipped = 1 if targets in selector.board_combos[boardname]['wordpair'] else 0
    target_idx = list(selector.board_combos[boardname]['wordpair']).index(targets)# if unflipped else list(selector.board_combos[boardname]['wordpair']).index(flipped_targets)
    fit = selector.fit(boardname, target_idx)
    diag = selector.diagnosticity(boardname, target_idx)
    print("diag", diag)
    inf = selector.informativity(boardname, target_idx)
    cost = selector.cost[cost_fn_fit][targets] 
    utility = selector.pragmatic_speaker(targets, boardname, cost_fn_fit, range(len(selector.vocab)))
    #print('clue, fit, diagnositicity, cost, speaker prob')
    table_data = []
    for word in candidates:
      idx = list(selector.vocab["Word"]).index(word)
    #   print('word:', word)
    #   print('idx:', idx)
      #print(word, ':', fit[idx], diag[idx], cost[idx], utility[idx])
      #print(f"clue: {word}, fit: {fit[idx]}, diag: {diag[idx]}, cost: {cost[idx]}, utility: {utility[idx]}")
      # Add a row to the table_data list
      table_data.append([word, fit[idx], diag[idx], cost[idx], utility[idx]])
      #table_data.append([word, fit[idx], 0, cost[idx], utility[idx]])
      #print(f"clue: {word}, fit: {fit[idx]}, diag: 0, cost: {cost[idx]}, utility: {utility[idx]}")
      # distractor_pairs = np.delete(list(self.board_combos[boardname]['wordpair']), target_idx, axis=0)
      # distractors = np.delete(self.sims[boardname], target_idx, axis=0)
      # print("for clue", word)
      # print('biggest distractor', distractor_pairs[np.argmax(distractors,axis=0)[idx]])

    # also find the BEST clue based on the model parameters that is not the two target words
    # remove the two target words from utility and find the best in the vocab
    # Define headers for the table
    headers = ["Clue", "Fit", "Diag", "Cost", "Utility"]

    # Print the table using tabulate
    print(tabulate(table_data, headers=headers))
    # non_targets = [word for word in selector.vocab["Word"] if word not in targets.split('-')]
    # non_target_idx = [list(selector.vocab["Word"]).index(word) for word in non_targets]
    # non_target_utility = utility[non_target_idx]
    # best_clue = non_targets[np.argmax(non_target_utility)]
    # print('best clue:', best_clue)

    



    

if __name__ == "__main__":
  #exp_path = '../data/exp1/'
  #selector = Selector(exp_path, sys.argv[1:])
  #selector.print_examples()
  # out = selector.get_speaker_df()
  # out.to_csv(
  #   f'{exp_path}/model_output/speaker_df_{selector.cost_type}_{selector.inf_type}_{sys.argv[6]}.csv'
  # )

#   exp_path = '../data/exp3/'
#   selector = Selector(exp_path, sys.argv[1:])

#   # find optimal parameters
#   #selector.optimize('spearman')
#   out = selector.get_speaker_df()
#   out.to_csv(
#     f'{exp_path}/model_output/full_grid/speaker_df_{selector.cost_type}_{selector.inf_type}_{sys.argv[6]}.csv'
#   )

  # try out example
  # params is organized as [cost_type, inf_type, alpha, costweight, distweight]
  #print("experiment 1")

  # selector = Selector('../data/exp1/', ['cdf', 'RSA', 32, 0.16, 1])
  # selector.print_examples('lion-tiger', 256, ['cat', 'feline','animal','mammal'])
  # selector = Selector('../data/exp1/', ['cdf', 'RSA', 32, 0.20, 0.40])
  # selector.print_examples('snake-ash', 1024, ['poison', 'death', 'burn', 'hiss'])
  #selector = Selector('../data/exp1/', ['cdf', 'RSA', 28, 0.10, 0.04])
  #selector.print_examples('quick-glow', 1024, ['light', 'firework', 'fire', 'flash'])
  #selector.print_examples('lion-tiger', 256, ['animal','mammal', 'roar','cat', 'feline', 'bear', 'cheetah', 'jaguar', 'predator', 'wild', 'zoo'])
#   print("experiment 2: BEAR on board")
#   selector = Selector('../data/exp2/', ['cdf', 'RSA', 10, 0.4, 1])
#   selector.print_examples('lion-tiger', 256, ['animal','mammal', 'roar','cat', 'feline', 'bear', 'cheetah', 'jaguar', 'predator', 'wild', 'zoo'])
#   print("experiment 4: NO BOARD")
#   selector = Selector('../data/exp4/', ['cdf', 'additive', 20, 0.32, 0.64])
#   selector.print_examples('lion-tiger', 1024, ['animal','mammal', 'roar','cat', 'feline', 'bear', 'cheetah', 'jaguar', 'predator', 'wild', 'zoo'])
  #print("experiment 3: ENDORSEMENTS")
  #selector = Selector('../data/exp3/', ['cdf', 'RSA', 4, 0.32, 0.40])
  #selector.print_examples('lion-tiger', 2048, ['cat', 'feline','animal','mammal'])
  # selector = Selector('../data/exp3/', ['cdf', 'RSA', 8, 0.02, 0.20])
  # selector.print_examples('snake-ash', 1024, ['poison', 'death', 'burn', 'hiss'])
  #selector = Selector('../data/exp3/', ['cdf', 'RSA', 32, 0.20, 0.12])
  #selector.print_examples('quick-glow', 1024, ['light', 'firework', 'fire', 'flash'])
  
  print("experiment ASSOCIATIONS")
  selector = Selector('../data/exp4/', ['cdf', 'RSA', 4, 0.32, 0.70])
  selector.print_examples('lion-tiger', 2048, ['cat', 'animal', 'mammal', 'feline'])