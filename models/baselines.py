import pandas as pd
import numpy as np
import math
import scipy
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

class Baseline:
    '''
    Class executing random walks from association norms given by the SWOW model.
    '''
    def __init__(self) :
        self.vocab = list(pd.read_csv("../data/exp1/model_input/vocab.csv").Word)

    def midpoint_scores(self, w1, w2):
        # import swow associative embeddings
        embeddings = pd.read_csv("../data/exp1/model_input/swow_associative_embeddings.csv").transpose().values
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
            clue_list = group['correctedClue'].to_numpy()
            for clue in clue_list:
                index = closest_to_midpoint.index(clue) if clue in closest_to_midpoint else None
                scores['mid_index'].append(index)
            rows.append(group)

        pd.concat(
            [
                pd.concat(rows,axis=0,ignore_index=True),
                pd.DataFrame.from_dict(scores)
            ],
            axis=1
        ).to_csv('/'.join(data_path.split('/')[:-1])+'/model_output/midpoint_scores.csv')

    def save_frequency_scores(self,data_path):
        expdata = pd.read_csv(f"{data_path}", encoding= 'unicode_escape')
        scores = defaultdict(list)
        rows = []
        freq = list(pd.read_csv("../data/exp1/model_input/vocab.csv").sort_values(by="LgSUBTLWF", ascending=False)["Word"])
        # sort by frequency
        for name, group in expdata.groupby('wordpair') :
            clue_list = group['correctedClue'].to_numpy()
            for clue in clue_list:
                index = freq.index(clue) if clue in freq else None
                scores['freq_index'].append(index)
            rows.append(group)

        pd.concat(
            [
                pd.concat(rows,axis=0,ignore_index=True),
                pd.DataFrame.from_dict(scores)
            ],
            axis=1
        ).to_csv('/'.join(data_path.split('/')[:-1])+'/model_output/freq_scores.csv')
      
    def shared_neighbors(self, w1, w2, alpha):
      # Read the strengths file
      df = pd.read_csv("../data/walk_data/swow_strengths.csv")

      # Find all neighbors of word1
      neighbors_word1 = set(df[df['cue'] == w1]['response'].tolist())
      
      
      # Find all neighbors of word2
      neighbors_word2 = set(df[df['cue'] == w2]['response'].tolist())
      

      strength_dict = {}
      freq_dict = {}

      for word in self.vocab:
          strength_to_word1 = df[(df['cue'] == w1) & (df['response'] == word)]['R123.Strength'].values[0] if word in neighbors_word1 else 0.0001
          strength_to_word2 = df[(df['cue'] == w2) & (df['response'] == word)]['R123.Strength'].values[0] if word in neighbors_word2 else 0.0001
          
          combined_strength = strength_to_word1 * strength_to_word2

          freqs = pd.read_csv("../data/exp1/model_input/vocab.csv")


          log_freq = freqs[freqs['Word'] == word]['LgSUBTLWF'].values[0] if word in freqs['Word'].tolist() else 0.0001
          
          
          strength_dict[word] = combined_strength
          freq_dict[word] = log_freq

      # z-score dicts

      strength_dict = {k: v for k, v in zip(strength_dict.keys(), MinMaxScaler().fit_transform(np.array(list(strength_dict.values())).reshape(-1, 1)).flatten())}
      freq_dict = {k: v for k, v in zip(freq_dict.keys(), MinMaxScaler().fit_transform(np.array(list(freq_dict.values())).reshape(-1, 1)).flatten())}

      combined_scores_dict = {}
      for word in self.vocab:
          # Calculate combined score
          combined_score = alpha * strength_dict[word] + (1 - alpha) * freq_dict[word]
          combined_scores_dict[word] = combined_score

      # z-score this combined score dict using MinMaxScaler
      combined_scores_dict = {k: v for k, v in zip(combined_scores_dict.keys(), MinMaxScaler().fit_transform(np.array(list(combined_scores_dict.values())).reshape(-1, 1)).flatten())}    

      
      # sort by combined score in descending order and return the keys
      combined_scores_dict = {k: v for k, v in sorted(combined_scores_dict.items(), key=lambda item: item[1], reverse=True)}
      return list(combined_scores_dict.keys())
    
    def save_mixture_scores(self, data_path):
      # import empirical clues (cleaned)
      expdata = pd.read_csv(f"{data_path}", encoding= 'unicode_escape')
      scores = defaultdict(list)
      alpha_dict = defaultdict(list)
      rows = []
      for a in np.arange(0, 1.1, 0.1):
        print("alpha = ", a)
        for name, group in expdata.groupby('wordpair') :
              print("wordpair = ", name)
              vocab_scores = self.shared_neighbors(group['Word1'].to_numpy()[0], group['Word2'].to_numpy()[0], alpha = a)              
              clue_list = group['correctedClue'].to_numpy()
              for clue in clue_list:
                  index = vocab_scores.index(clue) if clue in vocab_scores else None
                  scores['mixture_index'].append(index)
                  alpha_dict['alpha'].append(a)
              rows.append(group)

      pd.concat(
          [
              pd.concat(rows,axis=0,ignore_index=True),
              pd.DataFrame.from_dict(scores),
              pd.DataFrame.from_dict(alpha_dict)
          ],
          axis=1
      ).to_csv('/'.join(data_path.split('/')[:-1])+'/model_output/mixture_scores.csv')

class nonRSA:

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

class utils:
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



if __name__ == "__main__" :
    baseline = Baseline()
    #baseline.save_midpoint_scores('../data/exp1/cleaned.csv')
    #baseline.save_frequency_scores('../data/exp1/cleaned.csv')
    baseline.save_mixture_scores('../data/exp1/cleaned.csv')
