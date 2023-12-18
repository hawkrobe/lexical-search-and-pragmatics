import json
import pickle
import itertools
import warnings
import scipy.spatial.distance

import pandas as pd
import numpy as np
import networkx as nx

warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.special import softmax

class RSA:
  def __init__(self, exp_path) :
    self.exp_path = exp_path
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
    self.cost = {}
    #measure_df = pd.read_csv(f"{self.exp_path}/model_output/ranks_long.csv")
    #measure_df = pd.read_csv(f"{self.exp_path}/model_output/scores_long.csv")
    measure_df = pd.read_csv(f"{self.exp_path}/model_output/cdfs_long.csv")
    for measure in measure_df['measure'].unique() :
      subset = measure_df.query("measure == @measure")
      print(subset.query("wordpair == 'void-couch'"))
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

    ## next we find the product of similarities between c-w1 and c-w2 for that
    ## specific board's 190 word-pairs this gives us a 190 x N array of product
    ## similarities for a given combs_df specifically, for each possible pair...
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
    inputs:
    (1) board name ("e1_board1_words")
    (2) alpha: optimized parameter
    (3) costweight: optimized weight to freequency
    (4) modelname: 'glove'

    outputs:
    softmax likelihood of each possible clue
    '''
    targetpair_idx = list(self.board_combos[boardname]['wordpair']).index(targetpair)
    inf = np.log(self.literal_guesser(boardname))[targetpair_idx].ravel()
    cost = np.log1p(self.cost[modelname][targetpair]['value'].to_numpy()).ravel()
    utility = (1-costweight) * inf - costweight * cost
    return softmax(alpha * utility) # 190

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
    speaker_prob = []
    speaker_rank = []
    for index, row in boarddata.iterrows():
        clue1 = row["correctedClue"]
        wordpair = str(row["wordpair"]).replace(" ", "")

        # find index of clue
        if clue1 in list(self.vocab["Word"]):
            clue_index = list(self.vocab["Word"]).index(clue1)
            clue_probs = probsarray[clue_index]
            clue_rank = np.nonzero(probsarray_sorted==clue_index)[0][0]
        else:
            clue_rank = "NA"
            clue_probs = "NA"

        speaker_prob.append(clue_probs)
        speaker_rank.append(clue_rank)
    return speaker_prob, speaker_rank

  def get_speaker_df(self):
    '''
    returns a complete dataframe of pragmatic speaker ranks & probabilities over different representations
    over a given set of candidates

    inputs:
    params: optimal parameter dictionary for different representation keys

    outputs:
    a dataframe with clue ranks & probs for each possible candidate & representation
    '''
    speakerprobs_dfs = []
    for modelname in self.cost.keys() :
      for costweight in [0, 0.001, 0.01, 0.05, 0.5, 0.8, 1] :
        for alpha in [1, 5, 10, 15, 20, 25] :
          print(costweight, alpha)
          for index, row in self.targets.iterrows():
            boardname = row["boardnames"]
            targetpair = row['wordpair']
            y = self.pragmatic_speaker(targetpair, boardname, modelname, alpha, costweight)
            y_sorted = np.argsort(-y)
            expdata_board = self.cluedata.query("wordpair == @targetpair and boardnames == @boardname").copy()
            speaker_prob, speaker_rank = self.get_speaker_scores(expdata_board, y, y_sorted)
            expdata_board.loc[:,"cost"] = modelname
            expdata_board.loc[:,"alpha"] = alpha
            expdata_board.loc[:,"costweight"] = costweight
            expdata_board.loc[:,"prag_speaker_probs"] = speaker_prob
            expdata_board.loc[:,"prag_speaker_rank"] = speaker_rank
            speakerprobs_dfs.append(expdata_board)

    pd.concat(speakerprobs_dfs).to_csv(
      f'{self.exp_path}/model_output/speaker_df.csv'
    )

if __name__ == "__main__":
  rsa = RSA('../data/exp2/')
  rsa.get_speaker_df()
