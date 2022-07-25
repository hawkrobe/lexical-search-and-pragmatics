import random
import json
import heapq
import json
import pickle
import itertools
import sys
import walker
import math
import warnings
import scipy.spatial.distance
import operator

import pandas as pd
import numpy as np
import networkx as nx

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import MinMaxScaler, normalize
from numpy.linalg import matrix_power
from numpy.random import randint, choice
from scipy import stats
from sklearn import preprocessing
from scipy.special import softmax


class RSA:

  def compute_board_combos(board_name, boards):
    '''
    inputs:
    (1) board_name ("e1_board1_words")
    output:
    all pairwise combinations of the words on the board

    '''
    board = boards[board_name]
    all_possible_combs = list(itertools.combinations(board, 2))
    combs_df = pd.DataFrame(all_possible_combs, columns =['Word1', 'Word2'])
    combs_df["wordpair"] = combs_df["Word1"] + '-'+ combs_df["Word2"]
    return combs_df

  def get_wordpair_list(board_combos, board_name) :
    '''
    inputs:
    (1) board_combos from compute_board_combos
    (2) board__name ("e1_board1_words")
    output:
    a list of all wordpairs for a given board

    '''
    return list(board_combos[board_name]['wordpair'])

  def create_board_matrix(combs_df, context_board, representations, modelname, vocab, candidates):
    '''
    inputs:
    (1) combs_df: all combination pairs from a given board
    (2) context_board: the specific board ("e1_board1_words")
    (3) representation: embedding space to consider, representations
    (4) modelname: 'glove'
    (5) the vocab over which computations are occurring
    (6) candidates over which the board matrix needs to be computed

    output:
    product similarities of given vocab to each wordpair
    '''
    # grab subset of words in given board and their corresponding glove vectors
    board_df = vocab[vocab['vocab_word'].isin(context_board)]
    board_word_indices = list(board_df.index)
    board_words = board_df["vocab_word"]
    board_vectors = representations[modelname][board_word_indices]

    # need to obtain embeddings of candidate set

    candidate_index = [list(vocab["vocab_word"]).index(w) for w in candidates]

    embeddings = representations[modelname][candidate_index]

    ## clue_sims is the similarity of ALL clues in full candidate space to EACH word on board (size 20)
    clue_sims = 1 - scipy.spatial.distance.cdist(board_vectors, embeddings, 'cosine')

    ## once we have the similarities of the clue to the words on the board
    ## we define a multiplicative function that maximizes these similarities
    board_df.reset_index(inplace = True)

    ## next we find the product of similarities between c-w1 and c-w2 for that specific board's 190 word-pairs
    ## this gives us a 190 x N array of product similarities for a given combs_df
    ## specifically, for each possible pair, pull out
    f_w1_list =  np.array([clue_sims[board_df[board_df["vocab_word"]==row["Word1"]].index.values[0]]
                          for  index, row in combs_df.iterrows()])
    f_w2_list =  np.array([clue_sims[board_df[board_df["vocab_word"]==row["Word2"]].index.values[0]]
                          for  index, row in combs_df.iterrows()])

    # result is of length 190 for the product of similarities (i.e. how similar each word i is to BOTH in pair)
    # note that cosine is in range [-1, 1] so we have to convert to [0,1] for this conjunction to be valid
    return ((f_w1_list + 1) /2) * ((f_w2_list + 1)/2)

  def literal_guesser(board_name, representations, modelname, candidates, vocab, boards):
    '''
    inputs are:
    (1) board name ("e1_board1_words"),
    (2) representation: embedding space to consider, representations
    (3) modelname: 'glove'
    (4) candidates (a list ['apple', 'mango'] etc.)

    output:
    softmax likelihood of different wordpairs under a given set of candidates

    '''

    board_combos = {board_name : RSA.compute_board_combos(board_name,boards) for board_name in boards.keys()}

    board_matrices = {
      key : {board_name :RSA.create_board_matrix(board_combos[board_name], boards[board_name], representations, modelname, vocab, candidates)
            for board_name in boards.keys()}
      for (key, embedding) in representations.items()
    }
    boardmatrix = board_matrices[modelname][board_name]
    return softmax(boardmatrix, axis=0)

  def pragmatic_speaker(board_name, beta, costweight, representations, modelname, candidates, vocab, boards):
    '''
    inputs:
    (1) board name ("e1_board1_words")
    (2) beta: optimized parameter
    (3) costweight: optimized weight to freequency
    (4) representation: embedding space to consider, representations
    (5) modelname: 'glove'
    (6) candidates (a list of words/clues to iterate over)
    (7) vocab
    (8) boards: imported json file

    outputs:
    softmax likelihood of each possible clue in "candidates"

    '''
    candidate_index = [list(vocab["vocab_word"]).index(w) for w in candidates]
    literal_guesser_prob = np.log(RSA.literal_guesser(board_name, representations, modelname, candidates, vocab, boards))
    clues_cost = -np.array([list(vocab["LgSUBTLWF"])[i] for i in candidate_index])
    utility = (1-costweight) * literal_guesser_prob - costweight * clues_cost
    return softmax(beta * utility, axis = 1)

  def pragmatic_guesser(board_name, beta, costweight, representations,modelname,candidates, vocab, boards):
    return softmax(np.log(RSA.pragmatic_speaker(board_name, beta, costweight, representations, modelname, candidates, vocab, boards)), axis = 0)

  def get_speaker_scores(cluedata, speaker_word_pairs, probsarray, probsarray_sorted, candidate_df) :
    '''
    takes a set of clues and word pairs, and computes the probability and rank of each clue
    inputs:
    (1) cluedata: a subset of expdata with relevant clues and wordpairs
    (2) speaker_word_pairs: the specific wordpairs for which probabilities need to be computed
    (3) probsarray: a 3 x candidates array of pragmatic speaker predictions
    (4) probsarray_sorted: a sorted 3 x candidates array of pragmatic speaker predictions
    (5) candidate_df: a vocab-like df of only candidates

    outputs:
    softmax probabilities and ranks for each candidate in cluedata
    '''
    speaker_prob = []
    speaker_rank = []
    for index, row in cluedata.iterrows():
        clue1 = row["Clue1"]
        wordpair = str(row["wordpair"]).replace(" ", "")
        wordpair_index = speaker_word_pairs.index(wordpair)

        # find index of clue
        if clue1 in list(candidate_df["vocab_word"]):
            clue_index = list(candidate_df["vocab_word"]).index(clue1)
            clue_probs = probsarray[wordpair_index, clue_index]
            clue_rank = np.nonzero(probsarray_sorted==clue_index)[1][wordpair_index]
        else:
            clue_rank = "NA"
            clue_probs = "NA"

        speaker_prob.append(clue_probs)
        speaker_rank.append(clue_rank)
    return speaker_prob, speaker_rank

  def get_speaker_df(representations, combined_boards_df,params, candidates, vocab, cluedata, board_combos, target_df, boards ):
    '''
    returns a complete dataframe of pragmatic speaker ranks & probabilities over different representations
    over a given set of candidates

    inputs:
    (1) representations space
    (2) combined_boards_df: dataframe with all board words
    (3) params: optimal parameter dictionary for different representation keys
    (4) candidates: the words over which the ranks/probs need to be computed
    (5) vocab :full vocabulary
    (6) cluedata: the empirical cluedata
    (7) board_combos: output of compute_board_combos
    (8) target_df: target_df with all the wordpairs for which values are needed
    (9) boards: imported json file

    outputs:
    a dataframe with clue ranks & probs for each possible candidate & representation
    '''
    speakerprobs_df = pd.DataFrame()
    for modelname in representations.keys() :

      optimal_params = params[modelname]
      print("optimal_params =", optimal_params)

      for index, row in combined_boards_df.iterrows():
        board = row["boardwords"]
        boardname = row["boardnames"]
        wordpairlist = RSA.get_wordpair_list(board_combos, boardname)
        speaker_word_pairs = target_df[(target_df["boardnames"] == row["boardnames"]) &
                                      (target_df["Experiment"] == row["Experiment"])]["wordpair"]
        speaker_word_pairs = list(speaker_word_pairs)
        speaker_df_new = pd.DataFrame({'wordpair': speaker_word_pairs})
        speaker_model = RSA.pragmatic_speaker(boardname, optimal_params[0], optimal_params[1],representations, modelname, candidates, vocab, boards)
        ## this is created at the BOARD level
        y = np.array([speaker_model[wordpairlist.index(wordpair)] for wordpair in speaker_word_pairs])
        y_sorted = np.argsort(-y)

        ## so y has 3 vectors of clue probabilities (the 3 pairs on this board)
        ## now we need to go into cluedata and score the probabilities for those specific clues
        expdata_board = cluedata[(cluedata["Board"] == row["Board"]) & (cluedata["Experiment"] == row["Experiment"]) & (cluedata["Clue1"].isin(candidates))]

        candidate_df = vocab[vocab["vocab_word"].isin(candidates)].reset_index()

        speaker_prob, speaker_rank = RSA.get_speaker_scores(expdata_board, speaker_word_pairs, y, y_sorted, candidate_df)
        expdata_board.loc[:,"representation"] = modelname
        expdata_board.loc[:,"prag_speaker_probs"] = speaker_prob
        expdata_board.loc[:,"prag_speaker_rank"] = speaker_rank
        speakerprobs_df = pd.concat([speakerprobs_df, expdata_board])

    return speakerprobs_df
  
  def get_RSA_union_int(union_dict, intersection_dict, target_df, boards, optimal_params, vocab, representations, expdata, resultspath):
    beta = optimal_params['swow'][0]
    cost = optimal_params['swow'][1]

    candidateprobs_RSA = pd.DataFrame()


    expdata_split = expdata.groupby('wordpair')
    budget_types = list(intersection_dict['happy-sad'].keys())

    for wordpair, group in expdata_split:
        boardname = group['boardnames'].iloc[0]
        
        target_main = target_df.loc[target_df['boardnames'] == boardname]
        target_main.reset_index(inplace = True)
        
        cluedict_union = union_dict[wordpair]
        cluedict_intersection = intersection_dict[wordpair]
        
        for budget in budget_types:
          wp_union = "NA"
          wp_intersection = "NA"
          cluelist_intersection = []
          cluelist_union = []
          
          if len(cluedict_union[budget]) > 0:
            cluelist_union = cluedict_union[budget]
            board_probs_union =  RSA.pragmatic_speaker(boardname, beta, cost, representations, 'swow', cluelist_union, vocab, boards)
            
            combos_df = RSA.compute_board_combos(boardname,boards)
            wordpairlist = list(combos_df["wordpair"])
            wp_union = board_probs_union[wordpairlist.index(wordpair)]
            

          if len(cluedict_intersection[budget]) > 0:
            cluelist_intersection = cluedict_intersection[budget]
            
            board_probs_intersection = RSA.pragmatic_speaker(boardname, beta, cost, representations, 'swow', cluelist_intersection, vocab, boards)
            combos_df = RSA.compute_board_combos(boardname,boards)
            wordpairlist = list(combos_df["wordpair"])
            wp_intersection = board_probs_intersection[wordpairlist.index(wordpair)]
                
          for index, row in group.iterrows():
            clue = row["correctedClue"]
            
            clue_index_intersection = cluelist_intersection.index(clue) if clue in cluelist_intersection else -1
            clue_index_union = cluelist_union.index(clue) if clue in cluelist_union else -1
            
            
            clue_prob_intersection = wp_intersection[clue_index_intersection] if clue_index_intersection != -1 else 0.000000001
            clue_prob_union = wp_union[clue_index_union] if clue_index_union != -1 else 0.000000001
            
            clue_board_df = pd.DataFrame({'wordpair': [wordpair]})
            clue_board_df["boardnames"] = boardname
            clue_board_df["budget"] = budget
            clue_board_df["clue"] = clue
            clue_board_df["clue_probs_union"] = clue_prob_union
            clue_board_df["clue_probs_intersection"] = clue_prob_intersection
            
            candidateprobs_RSA = pd.concat([candidateprobs_RSA, clue_board_df])
            candidateprobs_RSA.to_csv(resultspath, index=False)
    return candidateprobs_RSA


class nonRSA:
  def apply_corrections(df, corrections, vocab):
    df = df.copy()
    df["correctedClue"] = "NA"
    for index, row in df.iterrows():
        clue = row['Clue1']
        if clue not in list(vocab.vocab_word) and clue in list(corrections.vocab_word):
            df.at[index, 'correctedClue'] = corrections.loc[corrections.vocab_word == clue].correction.values[0]
        else:
            df.at[index, 'correctedClue'] = clue
    return df
  
  def get_distinctiveness(context_board, alpha, candidates, representations, modelname, vocab, target_df):
    '''
    takes in a given board, target wordpairs, and a set of clue candidates, and returns the distinctivess
    based on alpha(clue-w1*clue-w2) - (1-alpha)*(average of all other words on board)
    '''

    # grab subset of words in given board and their corresponding glove vectors
    board_df = vocab[vocab['vocab_word'].isin(context_board)]
    board_word_indices = list(board_df.index)
    board_vectors = representations[modelname][board_word_indices]

    ## clue_sims is the similarity of ALL clues in full searchspace (size N) to EACH word on board (size 20)

    ### NEED TO FIX THIS TO ONLY CONSIDER CANDIDATES!!
    candidate_index = [list(vocab["vocab_word"]).index(w) for w in candidates]
    candidate_embeddings = representations[modelname][candidate_index]

    clue_sims = (1-scipy.spatial.distance.cdist(board_vectors, candidate_embeddings, 'cosine') + 1 ) / 2
    target_sample = target_df[target_df['Word1'].isin(board_df["vocab_word"]) & target_df['Word2'].isin(board_df["vocab_word"])]
    w1_index = [list(board_df["vocab_word"]).index(row["Word1"]) for index, row in target_sample.iterrows()]
    w2_index = [list(board_df["vocab_word"]).index(row["Word2"]) for index, row in target_sample.iterrows()]
    clue_w1 = clue_sims[w1_index]
    clue_w2 = clue_sims[w2_index]
    clue_prod = np.multiply(clue_w1,clue_w2)

    # deleting the two target words to compute average similarity to other words on the board
    clue_sims_new = np.array([np.delete(clue_sims, [w1_index[i], w2_index[i]], axis=0) for i in range(len(w1_index))])
    avg_sim = np.mean(clue_sims_new, axis=1)

    ## FUNC = alpha(clue_w1*clue_w2) - (1-alpha)*(average of other board words)

    func = np.subtract((alpha)*clue_prod, (1-alpha)*avg_sim)
    return func

  def speaker_targetboard(context_board, alpha, beta, candidates, representations, modelname, vocab, target_df):
    '''
    takes in a given board, wordpairs, and a set of possible candidates, and returns the likelihood of
    each candidate for each target wordpair on that board based on:
    alpha(clue-w1*clue-w2) - (1-alpha)*(average of all other words on board)
    i.e., maximize similarity to wordpair and minimize similarity to other words

    inputs:
    (1) context_board: a specfic game board (e.g., boards['e1_board10_words'])
    (2) alpha: ranges from 0 to 1.1 in 0.1 increments
    (3) beta: tuning parameter
    (4) candidates: list of candidates to consider
    (5) representation: embedding space to consider, representations
    (6) modelname: 'glove'
    (7) vocab: search space over which likelihoods will be calculated
    (8) target_df: a dataframe that contains info about test wordpairs & which boards they come from

    output:
    softmax of clue likelihoods over specified candidates
    '''
    # grab subset of words in given board and their corresponding glove vectors
    board_df = vocab[vocab['vocab_word'].isin(context_board)]
    board_word_indices = list(board_df.index)
    board_words = board_df["vocab_word"]
    board_vectors = representations[modelname][board_word_indices]

    ## clue_sims is the similarity of ALL clues in full searchspace (size N) to EACH word on board (size 20)

    ### NEED TO FIX THIS TO ONLY CONSIDER CANDIDATES!!
    candidate_index = [list(vocab["vocab_word"]).index(w) for w in candidates]
    candidate_embeddings = representations[modelname][candidate_index]

    clue_sims = (1-scipy.spatial.distance.cdist(board_vectors, candidate_embeddings, 'cosine') + 1 ) / 2
    target_sample = target_df[target_df['Word1'].isin(board_df["vocab_word"]) & target_df['Word2'].isin(board_df["vocab_word"])]
    w1_index = [list(board_df["vocab_word"]).index(row["Word1"]) for index, row in target_sample.iterrows()]
    w2_index = [list(board_df["vocab_word"]).index(row["Word2"]) for index, row in target_sample.iterrows()]
    clue_w1 = clue_sims[w1_index]
    clue_w2 = clue_sims[w2_index]
    clue_prod = np.multiply(clue_w1,clue_w2)

    # deleting the two target words to compute average similarity to other words on the board
    clue_sims_new = np.array([np.delete(clue_sims, [w1_index[i], w2_index[i]], axis=0) for i in range(len(w1_index))])
    avg_sim = np.mean(clue_sims_new, axis=1)

    ## FUNC = alpha(clue_w1*clue_w2) + (1-alpha)*(average of other board words)

    func = np.subtract((alpha)*clue_prod, (1-alpha)*avg_sim)
    return softmax(beta * func, axis=1)
  
  def get_nonRSA_union_int(union_dict, intersection_dict, target_df, boards, optimal_params, vocab, representations, expdata, resultspath):
    candidateprobs_nonRSA = pd.read_csv('../data/exp3/nonRSAprobs.csv')
    
    expdata_split = expdata.groupby('wordpair')
    for wordpair, group in expdata_split:
      if wordpair not in candidateprobs_nonRSA.wordpair.values:
        boardname = group['boardnames'].iloc[0]
        target_main = target_df.loc[target_df['boardnames'] == boardname]
        target_main.reset_index(inplace = True)
        wordpair_index = target_main.index[(target_main['wordpair'] == wordpair)].tolist()[0]

        cluedict_union = union_dict[wordpair]
        cluedict_intersection = intersection_dict[wordpair]

        budget_types = list(intersection_dict['happy-sad'].keys())
        
        for alpha in np.arange(0,1.1, 0.1):
          for budget in budget_types:
            wp_union = "NA"
            wp_intersection = "NA"
            cluelist_intersection = []
            cluelist_union = []
            
            if len(cluedict_union[budget]) > 0:
              cluelist_union = cluedict_union[budget]
              board_probs_union = nonRSA.speaker_targetboard(boards[boardname], alpha, optimal_params['swow'][0], cluelist_union, representations, 'swow', vocab, target_df) 
              wp_union = board_probs_union[wordpair_index]

            if len(cluedict_intersection[budget]) > 0:
              
              cluelist_intersection = cluedict_intersection[budget]
              board_probs_intersection = nonRSA.speaker_targetboard(boards[boardname], alpha, optimal_params['swow'][0], cluelist_intersection, representations, 'swow', vocab, target_df)
              wp_intersection = board_probs_intersection[wordpair_index]

            for index, row in group.iterrows():
              clue = row["correctedClue"]
              # get index in cluelist_intersection
              clue_index_intersection = cluelist_intersection.index(clue) if clue in cluelist_intersection else -1
              clue_index_union = cluelist_union.index(clue) if clue in cluelist_union else -1
              
              # get the probability of this clue in this budget
              clue_prob_intersection = wp_intersection[clue_index_intersection] if clue_index_intersection != -1 else 0.000000001
              clue_prob_union = wp_union[clue_index_union] if clue_index_union != -1 else 0.000000001
                # add the clue probability to the dataframe      

              clue_board_df = pd.DataFrame({'wordpair': [wordpair]})
              clue_board_df["boardnames"] = boardname
              clue_board_df["alpha"] = alpha
              clue_board_df["budget"] = budget
              clue_board_df["clue"] = clue
              clue_board_df["clue_probs_union"] = clue_prob_union
              clue_board_df["clue_probs_intersection"] = clue_prob_intersection
              
              candidateprobs_nonRSA = pd.concat([candidateprobs_nonRSA, clue_board_df])
              candidateprobs_nonRSA.to_csv(resultspath, index = False)
            
    return candidateprobs_nonRSA
  
  def get_common_candidates(union_dict, intersection_dict, expdata, resultspath):
    common_candidates = pd.DataFrame()
    
    for index, row in expdata.iterrows():
      ID = row['clueGiverID']
    
      wordpair = row["wordpair_id"]
      w1, w2 = wordpair.split("-")
      reverse_wordpair = w2 + "-" + w1
      behavioral_clue_list = row["clue_list"]
      clueFinal = row["clueFinal"]
    
      cluedict_union = union_dict[wordpair] if wordpair in union_dict.keys() else union_dict[reverse_wordpair]
      cluedict_intersection = intersection_dict[wordpair] if wordpair in intersection_dict.keys() else intersection_dict[reverse_wordpair]
      
      budget_types = list(intersection_dict['happy-sad'].keys())
      
      for budget in budget_types:
        
        cluelist_intersection = []
        cluelist_union = []
        intersection_common =  []
        union_common = []
        finalclue_index_union = -1
        finalclue_index_intersection = -1
        
        if budget in cluedict_union:
          cluelist_union = cluedict_union[budget]
          union_common = list(set(behavioral_clue_list).intersection(cluelist_union))
          finalclue_index_union = cluelist_union.index(clueFinal) if clueFinal in cluelist_union else -1
          
        if budget in cluedict_intersection:
          cluelist_intersection = cluedict_intersection[budget]
          intersection_common = list(set(behavioral_clue_list).intersection(cluelist_intersection))
          finalclue_index_intersection = cluelist_intersection.index(clueFinal) if clueFinal in cluelist_intersection else -1
          

        common_df = pd.DataFrame({'clueGiverID': [ID]})
        common_df["wordpair"] = wordpair
        common_df["Level"] = row["Level"]
        common_df["clueFinal"] = clueFinal
        common_df["budget"] = budget
        common_df["behavioral_clue_list"] = str(behavioral_clue_list)
        common_df["len_cluelist_behavioral"] = len(behavioral_clue_list)
        common_df["len_cluelist_union"] = len(cluelist_union)
        common_df["union_common"] = str(union_common)
        common_df["len_union_common"] = len(union_common)
        common_df["finalclue_index_union"] = finalclue_index_union
        common_df["len_cluelist_intersection"] = len(cluelist_intersection)
        common_df["intersection_common"] = str(intersection_common)
        common_df["len_intersection_common"] = len(intersection_common)
        common_df["finalclue_index_intersection"] = finalclue_index_intersection
          
        common_candidates = pd.concat([common_candidates, common_df])
        common_candidates.to_csv(resultspath, index = False)
    return common_candidates

  def speaker_targetboard_cluescores(modelnames, optimal_params, board_combos, boards, candidates, vocab, representations, target_df, cluedata):
    '''
    returns a dataframe of likelihoods of each possible clue in an input df, for different alpha values

    inputs:
    (1) modelnames = list of models ['glove', 'swow']
    (2) optimal parameter dictionary with optimal beta parameter for each modelname
    (3) board_combos: output of compute_board_combos
    (4) boards: the actual boards variable (input from json file)
    (5) candidates: list of candidates to consider
    (6) vocab: search space over which likelihoods will be calculated
    (7) representations: embedding space to consider, representations
    (8) target_df: a dataframe that contains info about test wordpairs & which boards they come from
    (9) cluedata: a df of each clue for which we want a likelihood score

    output:
    likelihood of each clue at different alpha levels for different modelnames
    '''

    target_df["wordpair"] = target_df["Word1"] + "-" + target_df["Word2"]

    clue_board_df_main = pd.DataFrame()

    for modelname in modelnames:
      for alpha in np.arange(0,1.1, 0.1):
        ## for a given alpha, compute the clue similarities at the board level
        beta = optimal_params[modelname][0]
        print(f"for {modelname} and alpha {alpha}")

        speaker_board_probs = {
            board_name : nonRSA.speaker_targetboard(boards[board_name], alpha, beta, candidates, representations, modelname, vocab, target_df)
            for board_name in boards.keys()
        }

        for board in speaker_board_probs.keys():

          ## get the clues we need scores for from expdatanew
          clue_main = cluedata.loc[cluedata['boardnames'] == board]
          target_main = target_df.loc[target_df['boardnames'] == board]

          target_main.reset_index(inplace = True)
          #print(target_main)

          for index, row in clue_main.iterrows():
            if row["Clue1"] in list(vocab["vocab_word"]):
              #print("clue is:", row["Clue1"])
              clue_index = list(vocab["vocab_word"]).index(row["Clue1"])
              #print("clue_index:",clue_index)
              wordpair = row["wordpair"]
              ## need to figure out specific wordpair this clue corresponds to
              wordpair_index = target_main.index[(target_main['wordpair'] == wordpair)].tolist()[0]
              #print("wordpair_index:",wordpair_index)
              # get a sorted array of the clue scores
              mainscores = speaker_board_probs[board][wordpair_index]
              sorted_clue_probs = np.argsort(-mainscores).tolist()
              #print("sorted_clue_probs_indices = ", sorted_clue_probs)

              # we next obtain the score for each clue for a specific wordpair
              clue_similarity = speaker_board_probs[board][wordpair_index][clue_index]
              # want to find index of this particular clue in the overall distribution
              clue_rank = sorted_clue_probs.index(clue_index)
              #print("clue_rank:",clue_rank)
            else:
              clue_similarity = "NA"
              clue_rank = "NA"

            clue_board_df = pd.DataFrame({'boardnames': [board]})
            clue_board_df["wordpair"] = wordpair
            clue_board_df["Clue1"] = row["Clue1"]
            clue_board_df["clue_score"] = clue_similarity
            clue_board_df["clue_rank"] = clue_rank
            clue_board_df["alpha"] = alpha
            clue_board_df["Model"] = modelname

            clue_board_df_main = pd.concat([clue_board_df_main, clue_board_df])

    return clue_board_df_main
