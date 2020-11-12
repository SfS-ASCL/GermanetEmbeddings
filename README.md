# GermanetEmbeddings

## General Description
This repository contains the code to retrofit any static pretrained embeddings to a thesaurus, e.g GermaNet. 

The idea is the following:

In general for static embeddings, each word is associated with a vector ( = word embedding). These vectors can be used to train a neural network, e.g. given the vectors of a bunch a words, predict whether the word is a noun (or anything you want the model to learn). That means that for any 'old' pretrained word embeddings (e.g. word2vec, glove, fasttext), every word has only one vector, even though the words have different senses (e.g. 'bank' will have one vector that will be close to 'finance', 'river'...). The retrofitting algorithm (https://arxiv.org/pdf/1411.4166.pdf) takes in any type of pretrained word embeddings and a dictionary which for each word (or more explicitly for each lexical unit) in the semantic lexicon contains a list of related words (or lexical units). It starts initializing every sense of a word (e.g. bank_1(finance), bank_2(river bank)) with the pretrained vector for 'bank' and then updates all embeddings such that the embedding for bank_1 will become more similar to 'money', 'finance' etc, and the embedding for bank_2(river bank) more similar to 'water', 'river' ...

## setup

To run the code you need a virtuel enviroment based on python3 with the following pip packages:

- germanetpy (if you want to use GermaNet) (0.2.1)
- tqdm (4.51.0)
- numpy (1.19.4)
- pandas (1.1.4)
- finalfusion (0.7.1)
- ffp (0.2.0)
- torch (1.7.0)
- sklearn (0.0)

## usage
if you need to construct a dictionary file first and you want to create it from GermaNet, you need to run 'prepare_germanet.py'

```
python prepare_germanet.py path_to_the_germanet_data path_to_where_to_store_dictionary \t --hypernyms --hyponyms
```

if you want to retrofit embeddings you need to run

```
python retrofit.py path_to_the_dictionary_created_from_thesaurus \t path_to_pretrained_embeddings path_to_where_to_store_the_new_embeddings
```

## retrofitted embeddings
in "embeddings" you can find GermaNet embedding retrofitted from pretrained embeddings from fasttext. every lexical unit in GermaNet is associated with an embedding. the vocab looks like this:

Palmbaum_l179661

so it is word_'l'lexicalunitid

an example thesaurusfile looks like this:
Dattelpalme_l64985	Palmengewächs_l64982;Palme_l64983;Industriepflanze_l66505;Nutzpflanze_l66506;Palmbaum_l179661;
Sagopalme_l64986	Palmengewächs_l64982;Palme_l64983;Industriepflanze_l66505;Nutzpflanze_l66506;Palmbaum_l179661;
Ölpalme_l64987	Palmengewächs_l64982;Palme_l64983;Industriepflanze_l66505;Nutzpflanze_l66506;Palmbaum_l179661;
