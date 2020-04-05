# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:00:53 2020

@author: Smitkumar Contractor
"""

# nltk for corpus - dataset
import nltk
from nltk.corpus import movie_reviews
# to get randomization
import random
# to create freq list
from collections import Counter
# to take log of prior and likelyhood
from math import log

def train_naive_bayes(count_train_docs, train_docs, C):
    """
    Trains our naive bayes model

    trains naive bayes model with multinomial distribution

    Parameters:
    count_train_docs (int) : number of documents in training set
    train_docs (list of tupels) : each tuple in list has a words of a doc and
                                    doc's label
    C (list of stirngs) : list of different classes

    Returns:
    log_prior (dict) : holds prior for each class
    log_likelyhood (dict) : holds P(w|class) for each word in vocabulary
    vocab (set) : holds vocabulary

    """
    log_likelyhood = {}
    class_count_docs = {}
    class_docs = {}
    log_prior = {}
    vocab = set()
    # to hold counts of pos/neg docs and create big_doc for pos/neg
    for i in range(count_train_docs):
        # add current documents words to the vocabulary
        for w in train_docs[i][0]:
            vocab.add(w)
        # increase count of class and add words to big_doc for pos/neg
        if train_docs[i][1] in class_count_docs.keys():
            class_count_docs[train_docs[i][1]] += 1
            class_docs[train_docs[i][1]] += train_docs[i][0]
        else:
            class_count_docs[train_docs[i][1]] = 1
            class_docs[train_docs[i][1]] = train_docs[i][0]
    # get vocabulary's length
    vocab_len = len(vocab)
    # compute P(c) amd P(w|c) for each word for all classes in C
    for c in C:
        # compute P(c)
        log_prior[c] = log(class_count_docs[c] / float(count_train_docs))
        # frequency for big doc of current class
        freq_words_c = Counter(class_docs[c])
        # sum up every vocabulary word's appearance in this big doc 
        cnt = 0
        for key in vocab:
            cnt += freq_words_c[key]
        # get the likelyhood
        for key in vocab:
            log_likelyhood[(key, c)] = log((freq_words_c[key] + 1) / 
                                           (cnt + vocab_len))
    return log_prior, log_likelyhood, vocab

def test_naive_bayes(test_doc, log_prior, log_likelyhood, C, V):
    """
    Get test score Cnb of each class for given doc
    
    tets given doc and return Cnb for each class in C

    Parameters:
    test_doc (tupel) : words of a doc and doc's label
    train_docs (list of tupels) : each tuple in list has a words of a doc and
                                    doc's label
    log_prior (dict) : holds prior for each class
    log_likelyhood (dict) : holds P(w|class) for each word in vocabulary
    C (list of stirngs) : list of different classes
    V (set) : vocabulary

    Returns:
    total_sum (dict) : holds Cnb for each class

    """
    total_sum = {}
    for c in C:
        total_sum[c] = log_prior[c]
        for w in test_doc[0]:
            if w in V:
                total_sum[c] += log_likelyhood[(w, c)]
    return total_sum

# holds all documents
documents = []
# count of total documents
count_total_docs = 0

# get the data in documents from corpus
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))
        count_total_docs += 1

# shuffle document
random.shuffle(documents)

# number of documents in train set and test set
count_train_docs = int(0.8 * count_total_docs)
count_test_docs = count_total_docs - count_train_docs

# create test set and train set
test_docs = documents[count_train_docs:]
train_docs = documents[:count_train_docs]

# different class values
C = ['pos', 'neg']

# train naive bayes
log_prior, log_likelyhood, V = train_naive_bayes(count_train_docs,
                                                   train_docs, C)

# holds value to later calculate statistics
correct = 0
wrong = 0

# testing data
for i in range(count_test_docs):
    # get Cnb
    total_sum = test_naive_bayes(test_docs[i], log_prior, log_likelyhood, C, V)
    # get label with the maximum value and compare with result to update stats
    if max(total_sum, key=total_sum.get) == test_docs[i][1]:
        correct += 1
    else:
        wrong += 1

# stats
print("predicted correctly = ", correct)
print("predicted wrong = ", wrong)
print("accuracy = ", correct / (correct + wrong))

