import pandas as pd
import numpy as np
from tqdm import trange
import argparse
import finalfusion
import ffp
from sklearn import preprocessing
import torch
import torch.nn.functional as F


def read_embeddings(embedding_path):
    """
    This method reads embeddings from a file, format can either be fifu (finalfusion), bin (fasttest) or w2v (word2vec)
    :param embedding_path: the path to the file that stores the embeddings
    :return: the finalfusion embedding object
    """
    embeddings = None
    if embedding_path.endswith("fifu"):
        embeddings = finalfusion.load_finalfusion(embedding_path, mmap=True)
    elif embedding_path.endswith("bin"):
        embeddings = finalfusion.load_fasttext(embedding_path)
    elif embedding_path.endswith("w2v"):
        embeddings = finalfusion.load_word2vec(embedding_path)
    if not embeddings:
        print("attempt to read invalid embedding file")
    return embeddings


def initialize_vocabulary(thesaurus, embeddings, gn2general_map=None, add_general_words=False):
    """
    Given a dataframe with target words and related words, lookup pre-trained embeddings for each target word to
    construct
    a word2index and an embedding matrix
    :param thesaurus: a dataframe with two columns, the first column has to contain the target word, the second
    column the related words
    :param embeddings: a finalfusion embedding object
    :return: word2index: a dictionary with a target word and the corresponding index, embedding_matrix: at the
    correct index of each target word the pretrained embedding for it
    """
    column_names = list(thesaurus.columns.values)
    word2index = {}
    embedding_matrix = []
    i = 0
    pbar = trange(len(thesaurus), desc='construct vocab...', leave=True)
    for index, row in thesaurus.iterrows():
        pbar.update(1)
        target_word = row[column_names[0]].split(" ")[0]
        target_entry = row[column_names[0]]
        embedding = embeddings.embedding(target_word)
        if embedding is not None and target_entry not in word2index:
            word2index[target_entry] = i
            i += 1
            embedding_matrix.append(embedding)
    if add_general_words:
        for word in embeddings.vocab:
            index = embeddings.vocab.word_index[word]
            embedding = embeddings.storage[index]
            if embedding is not None and word not in word2index:
                word2index[word] = i
                i += 1
                embedding_matrix.append(embedding)
    embedding_matrix = np.array(embedding_matrix)
    if gn2general_map:
        replace_embedding_with_other_word(gn2word_map=gn2general_map, word2index=word2index,
                                          embedding_matrix=embedding_matrix, embeddings=embeddings)
    pbar.close()
    return word2index, embedding_matrix


def replace_embedding_with_other_word(gn2word_map, word2index, embedding_matrix, embeddings):
    """
    Replace the current key words of a given dictionary with the given value words (the embeddings)
    :param gn2word_map: the dictionary with "germanet_specific_word" as key and "general_word" as val
    :param word2index: the word2index map
    :param embedding_matrix: the current embedding matrix
    :param embeddings: an embedding object
    :return: the updated embedding matrix
    """
    for gn_word, general_word in gn2word_map.items():
        current_index = word2index[gn_word]
        new_embedding = embeddings.embedding(general_word)
        embedding_matrix[current_index] = new_embedding
    return embedding_matrix


def update_embedding_with_other_word(word2word_map, word2index, embedding_matrix):
    for target_word, new_word in word2word_map.items():
        embedding = None
        if target_word in word2index:
            embedding = embedding_matrix[word2index[target_word]]
        if "," in new_word:
            words = new_word.split(",")

            for w in words:
                w = w.strip()
                index = word2index[w]
                new_embedding = embedding_matrix[index]
                if embedding is not None:
                    embedding += new_embedding
                else:
                    embedding = new_embedding
        else:
            index = word2index[new_word]
            new_embedding = embedding_matrix[index]
            if embedding is not None:
                embedding += new_embedding
            else:
                embedding = new_embedding
        current_index = word2index[target_word]
        if embedding is not None:
            embedding_matrix[current_index] = embedding
    return embedding_matrix


def retrofit(word2index, pretrained_embeddings, thesaurus, num_iters, alpha=1, sum_beta=1.0, normalize=False):
    """
    This method takes an embedding matrix and a vocabulary as input. each word embedding for each word in the
    vocabulary
    will be updated to be more similar to all words related to that word, e.g. Fisch_1 und Tier_1 will be made more
    similar and Fisch_2 and Sternkreiszeichen.
    :param word2index: a map that maps the string for each word to the index
    :param pretrained_embeddings: a matrix with [V x embedding_dim] were each entry corresponds to an embedding for a
    word in V
    :param thesaurus: a map with a node of an ontology as key and a number of related nodes as value
    :param numIters: the number of times the whole embedding matrix gets updated
    :param alpha: the weight for the original word
    :param sum_beta: the weight for the related words
    :return: a new matrix with each entry corresponding to an embedding for a word in V, eventually being updated
    during retrofitting
    """
    retrofitted_embeddings = pretrained_embeddings.copy()
    count = 0
    column_names = list(thesaurus.columns.values)
    print(column_names)
    for it in range(num_iters):
        print("epoch %d" % it)
        pbar = trange(len(thesaurus), desc='retrofit embeddings...', leave=True)
        count = 0
        if normalize:
            retrofitted_embeddings = preprocessing.normalize(retrofitted_embeddings, norm='l1')
        # loop through every node in the ontology and retrieve the related nodes
        for index, row in thesaurus.iterrows():
            word = row[column_names[0]]
            related_words = row[column_names[1]]
            if type(related_words) == float:
                continue
            related_words = related_words.strip().split(";")
            related_words.pop()

            num_related = len(related_words)
            pbar.update(1)
            # no related, pass - use original embedding, as not other words are available to make this embedding
            # closer to
            if num_related == 0:
                continue
            else:
                # keep track of the number of words that gets actually retrofitted
                count += 1
                # loop over neighbours and add to neighbor vectors (sum of weight = 1)
                word_index = word2index[word]
                # weigh the amount of the original embedding by alpha
                new_vec = pretrained_embeddings[word_index] * alpha
                # this is the weight for the amount of the related embedding
                beta = sum_beta / num_related
                for related_word in related_words:
                    # retrieve the index of the related word
                    related_word_index = word2index[related_word]
                    # compute the new vector for the related word by combining the target word embedding and related
                    # word embedding
                    new_vec += retrofitted_embeddings[related_word_index] * beta
                    new_vec = new_vec / (alpha + sum_beta)
                retrofitted_embeddings[word_index] = new_vec / (alpha + sum_beta)
        pbar.close()

    print("Actual retrofitted words = %d \n new vectors done" % count)
    return retrofitted_embeddings


def save_new_embeddings(word2index, embedding_matrix, save_path, normalize=True):
    """
    Saves the new word embeddings as finalfusion. if specified, saves a normalized embedding matrix as well.
    :param word2index: a dictionary with each lexunit word as key and an index as value
    :param embedding_matrix: the matrix of the retrofittet embeddings
    :param save_path: a file to write the new embeddings to, the .fifu ending is automatically attached
    :param normalize: a boolean flag, if True the retrofitted embedding will be normalized to unit length and will be
    saved as well.
    """
    vocab = ffp.vocab.SimpleVocab(words=list(word2index.keys()), index=word2index)
    print(len(word2index))
    print(embedding_matrix.shape)
    if normalize:
        normalized_embedding_matrix = F.normalize(torch.from_numpy(np.array(embedding_matrix)), p=2, dim=1).numpy()
        normalized_storage = ffp.storage.NdArray(normalized_embedding_matrix)
        e_normalized = ffp.embeddings.Embeddings(vocab=vocab, storage=normalized_storage)
        e_normalized.write(save_path + "_normalized.fifu")
    storage = ffp.storage.NdArray(embedding_matrix)
    e = ffp.embeddings.Embeddings(vocab=vocab, storage=storage)
    e.write(save_path + ".fifu")


def read_thesaurus2embedding_word(path, sep):
    """
    This method reads in a map, that maps a word from the thesaurus file to a word that should be stored as a key in
    the embedding matrix. e.g. if you have the lexunit Haus_l2342 and you want it to be represented by 'Haus_1' this
    method
    maps the word in Germanet to a string that will be stored in the embedding matrix
    :param path:
    :return: a dictionary that contains the words from the thesaurus to their corresponding string representations in
    the
    embedding vocabulary
    """
    data = pd.read_csv(path, sep=sep)
    column_names = list(data.columns.values)
    thesaurus2word = dict(zip(list(data[column_names[0]]), list(data[column_names[1]])))
    return thesaurus2word


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("thesaurus_file", type=str,
                      help="The GermaNet (or similar Thesaurus) file. It is supposed to contain two columns, "
                           "seperated by a separator. The first column contains a string representing a lexical unit "
                           "(e.g. Haus_1 or Haus_123123) and the second " \
                           "column contains related words (e.g. hyponyms, synonyms), separated with semicolon")
    argp.add_argument("sep", type=str,
                      help="the separator symbol for the thesaurus file, e.g . ; if the columns are separated by "
                           "semicolon")
    argp.add_argument("embedding_file", type=str,
                      help="The path to the pretrained embeddings that should be retrofitted")
    argp.add_argument("save_path", type=str,
                      help="the file that should be used to write the new, retrofitted embeddings")
    argp.add_argument("--additional_map", type=str,
                      help="optional, a map that provides a preferred string for the original lexical unit words.")
    argp = argp.parse_args()
    embeddings = read_embeddings(argp.embedding_file)
    thesaurus = pd.read_csv(argp.thesaurus_file, sep=argp.sep)
    column_names = list(thesaurus.columns.values)
    if argp.additional_map:
        word2words = read_thesaurus2embedding_word(argp.additional_map, argp.sep)
        w2index, matrix = initialize_vocabulary(thesaurus=thesaurus, embeddings=embeddings,
                                                add_general_words=True, gn2general_map=word2words)
    else:
        w2index, matrix = initialize_vocabulary(thesaurus=thesaurus, embeddings=embeddings,
                                                add_general_words=False)

    print("the size of the vocabulary of the retrofitted embeddings will be %d" % len(w2index))

    new_matrix = retrofit(word2index=w2index, pretrained_embeddings=matrix, thesaurus=thesaurus, num_iters=15,
                          normalize=False)

    save_new_embeddings(word2index=w2index, embedding_matrix=new_matrix, save_path=argp.save_path)
