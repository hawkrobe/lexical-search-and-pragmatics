import pandas as pd
import numpy as np
import math
import scipy
from collections import defaultdict

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

if __name__ == "__main__" :
    baseline = Baseline()
    baseline.save_midpoint_scores('../data/exp1/cleaned.csv')
    baseline.save_frequency_scores('../data/exp1/cleaned.csv')
