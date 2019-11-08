import pickle
import sys
import json
import os

from gensim.corpora import Dictionary

def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))

if __name__ == '__main__':

    # check if dictionary exists, if not create from source
    if os.path.exists('bigram_dict.dict'):
        mydict_bi = Dictionary.load_from_text('bigram_dict.dict')
    else:
        df = pd.read_csv('preprocessed_text_with_bigrams.csv')

        bi_X = df.loc[500000:,'cleaned_text_bigrams']
        y = df.loc[500000:,'label']

        train_df = df.head(500000)
        X_bi = train_df['cleaned_text_bigrams']

        texts_bi = [ast.literal_eval(lis) for lis in X_bi.values.tolist()]

        mydict_bi = gensim.corpora.Dictionary(texts_bi)

    model = load_model('models/model_high_penalty_bi.h5')

    text = ' '.join([str(word) for word in sys.argv[1:]])

    # convert all texts to dictionary indices
    train_texts_indices = list(map(lambda x: texts_to_indices(x, mydict_bi), [text]))
    # pad or truncate the texts
    X = pad_sequences(train_texts_indices, maxlen=int(90))

    label = model.predict(X)

    print({'label': str(label)})

    with open('label_results.txt', 'w') as outfile:
        json.dump({'label': str(label)}, outfile)
