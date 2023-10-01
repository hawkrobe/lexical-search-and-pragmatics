#!/usr/bin/env python3

def check_words(data_file, column_name):
    '''
    Checks if words are in vocab and find a close replacement if a word is not in vocab

    Args:
      data_file: path to a csv file
      column_name: column name of words/clues we need to check

    Returns:
     new_datafile: a pandas dataframe with a column of words/clues that are in vocab
    '''
    # read in data file
    data_file = pd.read_csv(data_file)
    new_datafile = pd.DataFrame()
    vocab = list(pd.read_csv("../data/vocab.csv").Word)
    ## loading fasttext model
    print("loading fasttext model")
    start_time = time.time()
    model = api.load("fasttext-wiki-news-subwords-300")
    print("model loaded, total time", time.time() - start_time)

    # keep count of how many times a word is replaced
    count_corrections = 0
    for index, row in data_file.iterrows():
        if row[column_name] in vocab:
            # create new column that records if a correction was made
            # and duplicates column_name into new column
            row["corrected"] = "no"
            row["correctedClue"] = row[column_name]
            new_datafile = new_datafile.append(row)
        else:
            # remove spaces
            word = row[column_name].strip().replace(" ", "")
            # use difflib to find the closest word in vocab & compute semantic similarity score with original word
            closest_word = difflib.get_close_matches(word, vocab, n=1, cutoff=0.6)
            if closest_word and word in model and closest_word[0] in model:
                # compute similarity score between original word and closest word
                vectors = model[[word, closest_word[0]]]
                similarity_score = 1- distance.cosine(vectors[0], vectors[1])
                # if similarity score is greater than 0.3, replace word with closest word
                if similarity_score > 0.3:
                    row["corrected"] = "yes"
                    row["correctedClue"] = closest_word[0]
                    new_datafile = new_datafile.append(row)
                    count_corrections += 1
            else:
                # if no closest word is found or vector not in model, drop row
                continue
    return new_datafile, count_corrections
