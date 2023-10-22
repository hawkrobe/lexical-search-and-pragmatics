import pandas as pd
import urllib
import requests
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


def get_log_freq(item):
    '''
        Description:
            Obtains the log count for a single item from the Google Books Ngram Dataset (version 2)
        Args:
            (1) embeddings: path to the CSV file of semantic embeddings
            (2) item: the item for which log count is to be obtained
        Returns:
            log_count: the log count of the item
    '''
    new = item.replace("_", " ")
    encoded_query = urllib.parse.quote(new)
    params = {'corpus': 'eng-us', 'query': encoded_query, 'topk': 10, 'format': 'tsv'}
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items())
    response = requests.get('https://api.phrasefinder.io/search?' + params)

    response_flat = re.split('\n|\t',response.text)[:-1]
    response_table = pd.DataFrame(np.reshape(response_flat, newshape=(-1,7))).iloc[:,:2]
    response_table.columns = ['word','count']
    response_table['word'] = response_table['word'].apply(lambda x: re.sub('_0','', x))

    count = response_table['count'].astype(float).sum()
    log_count = np.log10(max(count, 0.0001))  # Set a minimum value to avoid log(0)
    return log_count


def shared_neighbors_walk(word1, word2, strengths_file, vocab_file, alpha):
    # Read the CSV file
    df = pd.read_csv(strengths_file)
    vocab = pd.read_csv(vocab_file)

    # Find all neighbors of word1
    neighbors_word1 = set(df[df['cue'] == word1]['response'].tolist())
    
    
    # Find all neighbors of word2
    neighbors_word2 = set(df[df['cue'] == word2]['response'].tolist())
    

    strength_dict = {}
    freq_dict = {}

    for word in list(vocab['Word']):
        strength_to_word1 = df[(df['cue'] == word1) & (df['response'] == word)]['R123.Strength'].values[0] if word in neighbors_word1 else 0.0001
        strength_to_word2 = df[(df['cue'] == word2) & (df['response'] == word)]['R123.Strength'].values[0] if word in neighbors_word2 else 0.0001
        
        combined_strength = strength_to_word1 * strength_to_word2
        #print("combined strength for ", word, " is ", combined_strength)
        log_freq = vocab[vocab['Word'] == word]['LgSUBTLWF'].values[0] if word in vocab['Word'].tolist() else 0.0001
        
        
        strength_dict[word] = combined_strength
        freq_dict[word] = log_freq

    # z-score dicts

    strength_dict = {k: v for k, v in zip(strength_dict.keys(), MinMaxScaler().fit_transform(np.array(list(strength_dict.values())).reshape(-1, 1)).flatten())}
    freq_dict = {k: v for k, v in zip(freq_dict.keys(), MinMaxScaler().fit_transform(np.array(list(freq_dict.values())).reshape(-1, 1)).flatten())}

    combined_scores_dict = {}
    for word in list(vocab['Word']):
        # Calculate combined score
        combined_score = alpha * strength_dict[word] + (1 - alpha) * freq_dict[word]
        combined_scores_dict[word] = combined_score

    # z-score this combined score dict using MinMaxScaler
    combined_scores_dict = {k: v for k, v in zip(combined_scores_dict.keys(), MinMaxScaler().fit_transform(np.array(list(combined_scores_dict.values())).reshape(-1, 1)).flatten())}    

    
    # sort by combined score in descending order
    combined_scores_dict = {k: v for k, v in sorted(combined_scores_dict.items(), key=lambda item: item[1], reverse=True)}

    # print top 10 words
    #print("Top 10 words:", list(combined_scores.keys())[:10])

    return combined_scores_dict
    
    

def get_clue_scores(data_path):
    '''
    Extract info for given clue and word-pair.

    Args: 
      w1, w2: word pair
      clues: array of clues generated from word pair  
    Returns: 
      (union score, intersection score) of word
    Computes and saves clue scores to scores.csv
    '''
    # import empirical clues (cleaned)
    expdata = pd.read_csv(f"{data_path}", encoding= 'unicode_escape')
    scores = pd.DataFrame()
    rows = []

    # look up how often each clue was visited
    for name, group in expdata.groupby('wordpair') :
        print("for word pair ", name)
        for a in np.arange(0, 1.1, 0.1):
            print("alpha = ", a)
            vocab_scores = shared_neighbors_walk(group['Word1'].to_numpy()[0], group['Word2'].to_numpy()[0], "../data/walk_data/swow_strengths.csv","../data/vocab.csv",  alpha = a)

            # obtain the scores for the specific clues

            clue_scores_dict = {i: vocab_scores[i] if i in vocab_scores.keys() else 0.0001 for i in group['correctedClue'].to_numpy()}
            

            # Create a DataFrame to hold the scores
            clue_scores_df = pd.DataFrame.from_dict(clue_scores_dict, orient='index', columns=['score']).reset_index().rename(columns={'index': 'correctedClue'})
            clue_scores_df['alpha'] = a  # Add 'alpha' column
            clue_scores_df['Word1'] = group['Word1'].values[0]  # Add 'word1' column
            clue_scores_df['Word2'] = group['Word2'].values[0]  # Add 'word2' column

            # append to scores DataFrame
            scores = pd.concat([scores, clue_scores_df], axis=0, ignore_index=True)
            scores.to_csv('../data/exp1/forage_clue_scores.csv', index=False)

# Example usage:
#shared_neighbors_walk("travel", "ankle", "../data/walk_data/swow_strengths.csv","../data/vocab.csv",  0.9)

get_clue_scores("../data/exp1/exp1-cleaned.csv")


