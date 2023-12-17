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
    self.targets = pd.read_csv(f"{exp_path}/model_input/targets.csv")
    self.cluedata = pd.read_csv(f"{exp_path}/finalClues.csv")
    with open(f'{exp_path}/model_input/boards.json', 'r') as json_file:
      self.boards = json.load(json_file)

    self.representations = {
      'swow' : pd.read_csv(f"{exp_path}/model_input/swow_embeddings.csv").transpose().values
    }

    print(self.representations['swow'].shape)

    self.create_board_combos()
    self.sims = {
      modelname : {boardname : self.create_sim_matrix(modelname, boardname)
                   for boardname in self.boards.keys()}
      for modelname in self.representations.keys()
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

  def create_sim_matrix(self, modelname, boardname):
    '''
    product similarities of given vocab to each wordpair
    '''

    combo_df = self.board_combos[boardname]
    context_board = self.boards[boardname]

    # grab subset of words in given board and their corresponding semantic reps
    board_df = self.vocab.query('Word in @context_board')
    board_vectors = self.representations[modelname][list(board_df.index)]
    embeddings = self.representations[modelname]

    ## clue_sims is the similarity of ALL clues in full candidate space to EACH word on board (size 20)
    clue_sims = 1 - scipy.spatial.distance.cdist(board_vectors, embeddings, 'cosine')

    ## next we find the product of similarities between c-w1 and c-w2 for that specific board's 190 word-pairs
    ## this gives us a 190 x N array of product similarities for a given combs_df
    ## specifically, for each possible pair, pull out
    board_df.reset_index(inplace = True)
    f_w1 = [clue_sims[board_df[board_df["Word"]==row["Word1"]].index.values[0]]
            for  index, row in combo_df.iterrows()]
    f_w2 = [clue_sims[board_df[board_df["Word"]==row["Word2"]].index.values[0]]
            for  index, row in combo_df.iterrows()]

    # result is of length 190 for the product of similarities (i.e. how similar each word i is to BOTH in pair)
    # note that cosine is in range [-1, 1] so we have to convert to [0,1] for this conjunction to be valid
    return ((np.array(f_w1) + 1) /2) * ((np.array(f_w2) + 1)/2)

  def cost(self, boardname, modelname) :
    return

  def literal_guesser(self, boardname, modelname):
    '''
    literal guesser probability over each wordpair
    '''
    return softmax(self.sims[modelname][boardname], axis = 0)

  def pragmatic_speaker(self, boardname, modelname, beta, costweight):
    '''
    inputs:
    (1) board name ("e1_board1_words")
    (2) beta: optimized parameter
    (3) costweight: optimized weight to freequency
    (4) modelname: 'glove'

    outputs:
    softmax likelihood of each possible clue
    '''
    literal_guesser_prob = np.log(self.literal_guesser(boardname, modelname))
    clues_cost = self.costs[modelname][boardname]
    utility = (1-costweight) * literal_guesser_prob - costweight * clues_cost
    return softmax(beta * utility, axis = 1)

  def pragmatic_guesser(self, boardname, modelname, beta, costweight):
    speaker_prob = np.log(self.pragmatic_speaker(boardname, modelname, beta, costweight))
    return softmax(speaker_prob, axis = 0)

  def get_speaker_scores(self, expdata, probsarray, probsarray_sorted) :
    '''
    takes a set of clues and word pairs, and computes the probability and rank of each clue
    inputs:
    (2) target_word_pair: the specific wordpairs for which probabilities need to be computed
    (3) probsarray: a candidates array of pragmatic speaker predictions
    (4) probsarray_sorted: a sorted candidates array of pragmatic speaker predictions

    outputs:
    softmax probabilities and ranks for each candidate in cluedata
    '''
    speaker_prob = []
    speaker_rank = []
    for index, row in expdata.iterrows():
        clue1 = row["correctedClue"]
        wordpair = str(row["wordpair"]).replace(" ", "")

        # find index of clue
        if clue1 in list(self.vocab["Word"]):
            clue_index = list(self.vocab["Word"]).index(clue1)
            wordpair_index = list(self.board_combos[row['boardnames']]['wordpair']).index(wordpair)
            clue_probs = probsarray[wordpair_index,clue_index]
            clue_rank = np.nonzero(probsarray_sorted==clue_index)[1][wordpair_index]
        else:
            clue_rank = "NA"
            clue_probs = "NA"

        speaker_prob.append(clue_probs)
        speaker_rank.append(clue_rank)
    return speaker_prob, speaker_rank

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
    for modelname in self.representations.keys() :
      optimal_params = params[modelname]
      print("optimal_params =", optimal_params)

      for index, row in self.targets.iterrows():
        print(index, row)
        boardname = row["boardnames"]
        target_wordpair = row['wordpair']
        board = self.boards[boardname]
        y = self.pragmatic_speaker(boardname, modelname, optimal_params[0], optimal_params[1])
        y_sorted = np.argsort(-y)
        expdata_board = self.cluedata.query("wordpair == @target_wordpair and boardnames == @boardname")
        speaker_prob, speaker_rank = self.get_speaker_scores(expdata_board, y, y_sorted)
        expdata_board["representation"] = modelname
        expdata_board["prag_speaker_probs"] = speaker_prob
        expdata_board["prag_speaker_rank"] = speaker_rank
        speakerprobs_dfs.append(expdata_board)

    pd.concat(speakerprobs_dfs).to_csv(
      f'{self.exp_path}/model_output/speaker_df.csv'
    )

if __name__ == "__main__":
  rsa = RSA('../data/exp2/')
  rsa.get_speaker_df({
    'swow' : (25.1522030761838, 0.03863169001849234),
    'glove' : (22.336514544537227, 0.039)
  })
