import argparse
import json
import os.path
import numpy as np
import string
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from operator import add
from math import log
from math import fsum

from pyspark import SparkContext
# from pyspark.sql import SparkSession


def book_to_terms(book):
    """
    This converts a book to a list of individual words
    """
    _, contents = book
    # contents.split() will generate a bunch of individual tokens. Each term (word)
    # in this list is then run through a *local* map that strips any remaining
    # whitespace off either side of the word and converts it to lowercase.
    words = list(map(lambda word: word.strip().lower(), contents.split()))
    return words

def tokenize_words(no_quot_words):
    new = []
    for item in no_quot_words:
        new.extend(item.split(" "))
    return new

def remove_stopwords(word_count):
    """
    This simply tests whether the term in question is listed among the
    stopwords broadcast array.
    """
    stopwords = SW.value  # Extract the list from the broadcast value.
    word, count = word_count

    # Remember: values corresponding to TRUE evaluations are retained (FALSE
    # are filtered out of the RDD), so you want this statement to evaluate to
    # TRUE for words you want to keep (i.e., words NOT in the stopwords list).
    return word not in stopwords

def remove_punctuation_from_end(word):
    """
    This removes punctuation at the begining and at the end of a word.
    """
    punctuation = PUNC.value
    if len(word)>0 and word[0] in punctuation:
        word = word[1:]
    if len(word)>0 and word[-1] in punctuation:
        word = word[:-1]
    return word

def check_punctuation(word):
    """
    This detects the punctuation if the words start with or end with punctuation,
    and removes it by remove_punctuation_from _end if so.
    """
    punctuation = PUNC.value
    while len(word)>0 and (word[0] in punctuation or word[-1] in punctuation):
        word = remove_punctuation_from_end(word)
    return word

def doc2vec(doc_tuple): #<- <docid> <content> <label>
    """
    This takes the same document tuple that is the output of wholeTextFiles,
    and parses out all the words for a single document, AND builds the
    document-specific count vectors for each word.
    """
    docid, content, label = doc_tuple
    # This is how we know what document we're in--i.e., what document
    # count to increment in the count array.
    label_list = LABELS.value
    document_list = DOCS.value
    doc_index = document_list.index(docid)

    # Generate a list of words and do a bunch of processing.
    no_quot_words = content.split("&quot")
    words = tokenize_words(no_quot_words)
    # words = book_to_terms(["junk", content])

    out_tuples = []
    N = len(document_list) # Denominator for TF-IDF.
    punctuation = PUNC.value
    stopwords = SW.value
    for w in words:
        # Enforce stopwords and minimum length.
        if w in stopwords or len(w) <= 1: continue
        w = check_punctuation(w)
        lancaster_stemmer = LancasterStemmer()
        w = lancaster_stemmer.stem(w)
        # Build the document-count vector.
        count_vector = []
        for i in range(N):
            count_vector.append([i, label_list[i], 0])
        # initialize [[<docid(0)>, <label>, <count(0)>],
        #             [<docid(1)>, <label>, <count(0)>],
        #             [<docid(2)>, <label>, <count(0)>],...]
        count_vector[doc_index][2] += 1

    # Build a list of (word, vector) tuples. I'm returning them all at
    # one time at the very end, but you could just as easily make use
    # of the "yield" keyword here instead to return them one-at-a-time.
        out_tuples.append([w, count_vector])
    # [<word> [[<docid(0)>, <label(0)>, <count>],
    #          [<docid(1)>, <label(1)>, <count>],
    #          [<docid(2)>, <label(0)>, <count>],...]]
    return out_tuples

def combine_by_doc(list_1,list_2):
    new_list = []
    for i in range(len(list_1)):
        new_list.append([list_1[i][0],list_1[i][1],list_1[i][2]+list_2[i][2]])
    return new_list

def get_things_out(x):
    count_list = x[1]
    list_to_return = []
    for i in range(len(count_list)):
        list_to_return.append([x[0],count_list[i]])
    return list_to_return

def wordSpec2docSpec(wordSpec_rdd):
    summed_wordSpec_rdd = wordSpec_rdd.reduceByKey(combine_by_doc)
    #should still be['word',[[<docid0>,<label0>,<count>],...,[<docidN>,<labelN>,<count>]]]
    #but no duplicate words

    rdd_flat = summed_wordSpec_rdd.flatMap(get_things_out)
    #rdd[[<'word0'>,[<docid0>,<label0>,<count>]],...,[<'word0'>,[<docidN>,<labelN>,<count>]],
    #    [<'word1'>,[<docid0>,<label0>,<count>]],...,[<'word1'>,[<docidN>,<labelN>,<count>]],
    #    ...]
    #pair word with each document count and flat everything out

    rdd_flat_release = rdd_flat.map(lambda x: (x[1][0],([x[1][1],[x[0],x[1][2]]])))
    #rdd[[<docid0>,[<label0>,<'word0'>,<count>]],...,[<docidN>,[<labelN>,<'word0'>,<count>]],
    #    [<docid0>,[<label0>,<'word1'>,<count>]],...,[<docidN>,[<labelN>,<'word1'>,<count>]],
    #    ...]
    #move everything into right place

    docid_rdd = rdd_flat_release.map(lambda x: ((x[0],x[1][0]),x[1][1:]))
    docid_label_rdd = docid_rdd.reduceByKey(lambda x,y: x+y)
#     docid_rdd = rdd_flat_release.reduceByKey(get_label_out)
    #rdd[[<docid0>,[<label0>,[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]],
    #    ...
    #    [<docidN>,[<label0>,[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]]
    #extracting the label
#     docid_label_rdd = []
#     docid_label_rdd = docid_rdd.map(lambda x: ((x[0],x[1][0]),x[1][1]))
    #rdd[[(<docid0>,<label0>),[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]],
    #    ...
    #    [(<docidN>,<label0>),[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]]
    #move the label out
    #just in case we need this version of data structure
    label_spec_rdd = docid_label_rdd.map(lambda x: (x[0][1],x[1]))
#     label_spec_rdd = docid_rdd.map(lambda x: (x[1][0],x[1][1]))
    #rdd[[<label0>,[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]],
    #    ...
    #    [<label0>,[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]]
    #move the label out and delete the document id
    return label_spec_rdd, docid_label_rdd

# def counts_to_tfidf(word_vector):
#   """
#   Computes the TF-IDF scores for each term-vector pair.
#   """
#   word, vector = word_vector
#   N = vector.shape[0]
#   nt = vector[vector > 0].shape[0] # Number of documents in which word appears.

#   # Compute IDF and TF-IDF.
#   idf = np.log(N / nt)
#   tfidf = np.array([tf * idf for tf in vector])
#   return (word, tfidf)

def laplace_smoothing(word_count_in_label, count_in_label):
    """
    This applys laplace smoothing to obtain the conditional probability of
    a word given specific label c. v is the total number of possible words in training set.
    """
    v = V.value
    nominator = word_count_in_label + 1
    denominator = count_in_label + v
    cond_prob = log(nominator / denominator)
    return cond_prob

def laplace_cond_prob(word_list):
    """
    This counts the conditional probability of each word, given the specific label,
    the probabilities are applied laplace smoothing.
    """
    total_count = 0
    for i in range(len(word_list)):
        total_count += word_list[i][1]
    for i in range(len(word_list)):
        word_list[i] = (word_list[i][0], laplace_smoothing(word_list[i][1], total_count))
        # word_list[i] = (word_list[i][0], word_list[i][1] / total_count)
    return word_list

def cond_prob_rdd(cp_rdd, rdd):
    """
    This changes former rdd with word counts into rdd with word conditional probabilities.
    The conditional probabilities here have already been applied laplace smoothing.
    """
    labels = rdd.map(lambda x: x[0]).distinct().collect()

    for ind in range(len(labels)):
        rdd_same_label = rdd.filter(lambda x: x[0]==labels[ind])
        rdd_label = rdd_same_label.map(lambda x: x[0]).distinct()

        list_same_label = rdd_same_label.flatMap(lambda x: tuple(x[1])).reduceByKey(add).collect()
        sum_count_in_label = rdd_label.map(lambda x: (x, list_same_label))
        rdd_cond_prob = sum_count_in_label.map(lambda x: (x[0], laplace_cond_prob(x[1])))
        cp_rdd = cp_rdd.union(rdd_cond_prob)
    return cp_rdd

def prior_prob_rdd(pp_rdd, rdd):
    """
    This changes former rdd grouped by document ids
    into rdd grouped by different labels.
    """
    doc_num = rdd.count()
    labels = rdd.map(lambda x: x[0]).distinct().collect()

    for ind in range(len(labels)):
        rdd_same_label = rdd.filter(lambda x: x[0]==labels[ind])
        rdd_label = rdd_same_label.map(lambda x: x[0]).distinct()

        doc_num_same_label = rdd_same_label.count()
        rdd_prior_prob = rdd_label.map(lambda x: (x, log(doc_num_same_label/doc_num)))
        pp_rdd = pp_rdd.union(rdd_prior_prob)
    return pp_rdd

# in rdd
def NBtraining(cp_rdd, pp_rdd):
    """
    This returns a huge rdd combined by cp_rdd and pp_rdd,
    and transforms each probability value.
    """
    # input: <label>, [(w1, cp1), (w2, cp2), ..., (wd,cpd)], pp
    # output: <label>, [(w1, cp1), (w2, cp2), ..., (wd,cpd), pp]
    rdd = cp_rdd.leftOuterJoin(pp_rdd)\
                .map(lambda x: (x[0], [x[1][0], x[1][1]]))
    return rdd

# need debug
def words_list(word_n_count_list):
    """
    This transfers list of word and its counts to a list that contains only words in each file.
    Each line of the rdd will be the word list of each document.
    """
    length = len(word_n_count_list)
    new_list = []

    for i in range(length):
        if not (word_n_count_list[i][1] == 0):
            new_list.append(word_n_count_list[i][0])

    # [["w_1", "w_2", "w_3", ......, "w_d1"],
    #  ["w_1", "w_2", "w_3", ......, "w_d2"], ...,
    #  ["w_1", "w_2", "w_3", ......, "w_dk"]]
    return new_list





def predict(train_list):
    """
    This determines the cond_prob of those words in both training and inputting sets
    and assigns them log(cond_prob), otherwise, cond_prob will use count 0 in laplace smoothing prob.
    """
    label, cp_list, pp = train_list[0], train_list[1][0:-1], train_list[1][-1]

    prob_list = []
    for words in range(len(cp_list)):
        if cp_list[words][0] in INPUT.value and cp_list[words][0] != 0:
            prob_list[words] = log(cp_list[words][1])
        else: prob_list[words] = log(laplace_smoothing(0, ?????))
    log_prob_sum = fsum(prob_list.append(pp))
    return (label, log_prob_sum)

def argmax(doc_prob_list):
    """
    This returns the label with the largest probability value in the list.
    """
    for l in range(len(doc_prob_list)):
        if doc_prob_list[l][1] == max(doc_prob_list[0:]):
            pred = doc_prob_list[l][0]
    return pred

def accurate_label_count(comparison_list):
    """
    This compares the prediction with the labels,
    and returns 1 if the prediction is in the label.
    """
    label, pred = comparison_list
    acc = 0
    if pred in label: acc = 1
    return ('acc', acc)

def accuracy(rdd_label, rdd_pred):
    """
    This calculates the accuracy by using count of correct prediction
    divided by the count of total amount of documents in inputting file.
    """
    rdd_accuracy = rdd_label.zip(rdd_pred)\
                            .map(accurate_label_count)\
                            .reduceByKey(add)\
                            .map(lambda x: x[1] / rdd_pred.count())
    return rdd_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CSCI 8360 Project 1",
        epilog = "answer key", add_help = "How to use",
        prog = "python p1.py [train-data] [train-label] [test-data] [optional args]")

    # Required args
    parser.add_argument("paths", nargs=3, #required = True
        help = "Paths of training-data, training-labels, and testing-data.")

    # Optional args
    # 	parser.add_argument("-s", "--stopwords", default = None.
    # 	        help = "Path to a file containing stopwords. [DEFAULT: None]")
    parser.add_argument("-a", "--algorithm", choices = ["NB", "LR"], default = "NB",
        help = "Algorithms to process classification: \"NB\": Naive Bayes, \"LR\": Logistic Regression [Default: Naive Bayes]")
    parser.add_argument("-o", "--output", default = ".",
        help = "Path to the output directory where outputs will be written. [Default: \".\"]")

    args = vars(parser.parse_args())
    sc = SparkContext()

    # Read in the variables
    training_data = args['paths'][0]
    training_label = args['paths'][1]
    testing_data = args['paths'][2]
    algorithm = args['algorithm']

    # Necessary Lists
    SW = sc.broadcast(stopwords.words('english'))
    PUNC = sc.broadcast(string.punctuation)

    # Generate RDDs of tuples
    rdd_train_data = sc.textFile(training_data)
    rdd_train_label = sc.textFile(training_label)
    rdd_test_data = sc.textFile(testing_data)
    rdd = rdd_train_data.zip(rdd_train_label)
    # <content> <label_list>

    # Preprocessing
    rdd = rdd.map(lambda x: (x[0], x[1].split(',')))

    valid_rdd = rdd.flatMapValues(lambda x: x)\
                    .filter(lambda x: 'CAT' in x[1])
    # <content> <label_containing_'CAT'_separated>

    valid_labels = valid_rdd.map(lambda x: x[1]).collect()
    LABELS = sc.broadcast(valid_labels)
    rdd_to_split = valid_rdd.zipWithIndex().map(lambda x: (x[1], x[0][0], x[0][1]))
    # <doc_id> <content> <label>

    #   doc_numb = rdd.count()
    #   DOCS = sc.broadcast(range(doc_numb))
    #   frequency_vectors = rdd.map(doc2vec)

    doc_index = rdd_to_split.map(lambda x: x[0]).collect()
    DOCS = sc.broadcast(doc_index)

    word_specific_frequency_vectors = rdd_to_split.flatMap(doc2vec)	#->not sure map or flatMap
    #Should look like
    #  rdd([['word0',[[<docid0>,<label0>,<count>],...,[<docidN>,<labelN>,<count>]]],
    #       ['word1',[[<docid0>,<label0>,<count>],...,[<docidN>,<labelN>,<count>]]],
    #                                                                       ...])
    doc_spec_frequency_vectors, with_id = wordSpec2docSpec(word_specific_frequency_vectors)
    #rdd[[<label0>,[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]],
    #    ...
    #    [<labelN>,[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]]

    #word list in each document (??)
    words_in_doc_rdd = with_id.map(lambda x: (x[0][0],words_list(x[1])))
#     words_in_doc_rdd_nodocid = words_in_doc_rdd.map(lambda x: x[1])

    words_in_label_rdd = doc_spec_frequency_vectors.map(lambda x: (x[0],words_list(x[1])))
    word_count_rdd = words_in_label_rdd.reduceByKey(lambda x,y: x+y)
    #word count in each label
    word_count_each_label_rdd = word_count_rdd.map(lambda x: (x[0],len(set(x[1]))))

    # Naive Bayes classifier
    word_numb = word_specific_frequency_vectors.count()
    #number of distinct words in training set
    V = sc.broadcast(word_numb)
    #broadcast number of distinct words in training set

    # model training
    cp_rdd = sc.parallelize([])
    cp_rdd = cond_prob_rdd(cp_rdd, doc_spec_frequency_vectors)
    pp_rdd = sc.parallelize([])
    pp_rdd = prior_prob_rdd(pp_rdd, doc_spec_frequency_vectors)
    rdd = NBtraining(cp_rdd, pp_rdd)

    # # input list
    # INPUT = sc.broadcast(words_list())
    # # prediction
    # prediction = rdd.map(predict).map(argmax)
    # # a list of predicted labels for input file
    # # ['label1', 'label2', 'label3', ..., 'labelk']

    # model testing
    # 1, training accuracy
    INPUT = sc.broadcast(words_list(rdd_train_data))
    rdd_train_pred = rdd.map(predict).map(argmax)
    rdd_train_acc = accuracy(rdd_train_label, rdd_train_pred)
    print(rdd_train_acc)
    # 2, testing accuracy
    INPUT = sc.broadcast(words_list(rdd_test_data))
    rdd_test_pred = rdd.map(predict).map(argmax)
    rdd_test_acc = accuracy(rdd_test_label, rdd_test_pred)
    print(rdd_test_acc)
