import argparse
import json
import os.path
import numpy as np
import string
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('wordnet')
from operator import add
from math import log

from pyspark import SparkContext

def tokenize_words(no_quot_words):
    """
    This tokenizes individual words
    """
    no_quot_words = no_quot_words.split("&quot") #.replace("--", " ")
    new = []
    for item in no_quot_words:
        new.extend(item.split(" "))
    return new

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

def cleanup_word(word):
    """
    This completes all the preprocessing required.
    Including removing punctuation and steming words
    """
    w = check_punctuation(word)
    # lancaster_stemmer = LancasterStemmer()
    # w = lancaster_stemmer.stem(w)
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    w = wnl.lemmatize(w.lower())
    # w = ps.stem(wnl.lemmatize(w.lower()))
    w = check_punctuation(w)
    return w

def cond_prob(word_count, ttl_count):
    v = V.value
    cond_prob = log((word_count+1) / (ttl_count+v))
    return cond_prob

def validation_format(rdd_file_in_textfile):
    # (doc_id, word)
    word_tokenize_rdd = rdd_file_in_textfile.map(lambda x: (x[0], tokenize_words(x[1])))
    words_rdd = word_tokenize_rdd.flatMapValues(lambda x: x).map(lambda x: (x[0], x[1]))
    clean_word_rdd = words_rdd.map(lambda x: (x[0], cleanup_word(x[1]))).filter(lambda x: len(x[1])>1 and x[1] not in SW.value)
    rdd_docid_word = clean_word_rdd.distinct().sortByKey()
    return rdd_docid_word

def fillna(cp, cp0):
    if cp == None: return cp0
    else: return cp

def predict(rdd_test_data, rdd_train):
    labels = LABELS.value
    rdd_train_labword_cp, rdd_train_lab_cp0, rdd_train_lab_pp = rdd_train[0], rdd_train[1], rdd_train[2]
    # (docid, label)
    rdd_test_docid = rdd_test_data.map(lambda x: x[0]) #document_ids
    rdd_test_doc_lab = rdd_test_docid.map(lambda x: (x, labels)).flatMapValues(lambda x: x)
    # (docid, word)
    rdd_test_doc_word = validation_format(rdd_test_data) #input_rdd_textfile
    # ((label, word), docid)
    rdd_test_labword_doc = rdd_test_doc_lab.leftOuterJoin(rdd_test_doc_word)\
                                .map(lambda x: ((x[1][0], x[1][1]), x[0])).sortBy(lambda x: x[1])
    # Prediction
    # ((label, word), (docid, cp))
    rdd_labword_doccp = rdd_test_labword_doc.leftOuterJoin(rdd_train_labword_cp)
    # (label, ((docid, cp), cp0))
    rdd_doclab_cpcp0 = rdd_labword_doccp.map(lambda x: (x[0][0], (x[1][0], x[1][1])))\
                                        .leftOuterJoin(rdd_train_lab_cp0)
    # ((doc, label), cp_fillNA)
    rdd_doclab_cp = rdd_doclab_cpcp0.map(lambda x: ((x[1][0][0], x[0]), fillna(x[1][0][1], x[1][1])))
    # ((doc, label), cp_sum)
    rdd_doclab_cpsum = rdd_doclab_cp.reduceByKey(add)
    # ((doc, label), (cp_sum, pp))
    rdd_doclab_pp = rdd_test_doc_lab.map(lambda x: (x[1], x[0])).leftOuterJoin(rdd_train_lab_pp)\
                                    .map(lambda x: ((x[1][0], x[0]), x[1][1]))
    rdd_doclab_cpsumpp = rdd_doclab_cpsum.leftOuterJoin(rdd_doclab_pp)
    # (doc, (label, cpsum+pp))
    rdd_doc_lablogp = rdd_doclab_cpsumpp.map(lambda x: (x[0][0], (x[0][1], sum(x[1]))))
    # (doc, pred)
    rdd_pred = rdd_doc_lablogp.map(lambda x: (x[0],(x[1][1],x[1][0])))\
                            .reduceByKey(max).sortByKey().map(lambda x: x[1][1])

    return rdd_pred.collect()

def cal_accuracy(label_list, pred_list):
    """
    This calculates the accuracy based on assigning 1 if the prediction of ith document
    in testing data is in the labels of ith document in testing label, and averaging
    the total counts.
    """
    cnt = 0
    ttl = len(label_list)
    for doc in range(ttl):
        if pred_list[doc] in label_list[doc]: cnt += 1
    accuracy = cnt / ttl
    return accuracy

def output_file(output_pred, output_path):
    outF = open(output_path, "w")
    textList = '\n'.join(output_pred)
    outF.writelines(textList)
    outF.close()
    return 'output_file has been saved!'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CSCI 8360 Project 1",
        epilog = "answer key", add_help = "How to use",
        prog = "python p1.py [train-data] [train-label] [test-data] [optional args]")

    # Required args
    parser.add_argument("paths", nargs=3, #required = True
        help = "Paths of training-data, training-labels, and testing-data.")

    # Optional args
    parser.add_argument("-s", "--size", choices = ["vsmall", "small", "large"], default = "vsmall",
        help = "Sizes to the selected file: \"vsmall\": very small, \"small\": small, \"large\": large [Default: \"vsmall\"]")
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

    # Necessary Lists
    SW = sc.broadcast(stopwords.words('english'))
    PUNC = sc.broadcast(string.punctuation)

    # Generate RDDs of tuples
    rdd_train_data = sc.textFile(training_data).zipWithIndex().map(lambda x: (x[1],x[0]))
    rdd_train_label = sc.textFile(training_label).zipWithIndex().map(lambda x: (x[1],x[0]))
    rdd_test_data = sc.textFile(testing_data).zipWithIndex().map(lambda x: (x[1],x[0]))
    # (content,label)
    rdd = rdd_train_data.join(rdd_train_label).map(lambda x: x[1])

    # Preprocessing --------------------------------------------------
    # Preprocessing to labels, leaving only ones with 'CAT'
    rdd = rdd.map(lambda x: (x[0], x[1].split(',')))
    # (label,content)
    rdd = rdd.flatMapValues(lambda x: x).filter(lambda x: 'CAT' in x[1]).map(lambda x: (x[1],x[0]))

    # total numb of all docs
    all_doc_numb = rdd.count()

    # Document Numbers for each label (RDD)
    doc_numb_in_label_rdd = rdd.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y)
    rdd = rdd.groupByKey().map(lambda x : (x[0], ' '.join(list(x[1])))).sortByKey(ascending=True)

    labels = rdd.map(lambda x: x[0]).collect()
    LABELS = sc.broadcast(labels)

    # (label,word)
    rdd = rdd.map(lambda x: (x[0], tokenize_words(x[1]))).flatMapValues(lambda x: x)
    # ((label, word),1)
    rdd = rdd.map(lambda x: ((x[0], cleanup_word(x[1])),1)).filter(lambda x: len(x[0][1])>1 and x[0][1] not in SW.value)
    # ((label, word), count)
    label_word_count_rdd = rdd.reduceByKey(lambda x,y: x+y)

    # (label, 0)
    labels_0 = sc.parallelize(labels).map(lambda x: (x,0)).collect()
    # all words in training
    words_in_training = label_word_count_rdd.map(lambda x: x[0][1]).distinct()
    # (word, [('CCAT',0),('MCAT',0),...])
    # (('CCAT', word),0)
    words_in_training_with_lab = words_in_training.map(lambda x: (x,labels_0)).flatMapValues(lambda x: x).map(lambda x: ((x[1][0],x[0]),x[1][1]))
    full_label_wct_rdd = label_word_count_rdd.union(words_in_training_with_lab).reduceByKey(lambda x,y: x+y)

    # (label,sum_count)
    word_count_label = full_label_wct_rdd.map(lambda x: (x[0][0],x[1])).reduceByKey(lambda x,y: x+y)
    # (label, (word,count))
    full_label_wct_rdd = full_label_wct_rdd.map(lambda x: (x[0][0],(x[0][1],x[1])))
    # (label, ((word,count),sum_count)) >>  ((label, word),(count,sum_count))
    full_label_wct_rdd = full_label_wct_rdd.leftOuterJoin(word_count_label).map(lambda x: ((x[0],x[1][0][0]),(x[1][0][1],x[1][1])))


    # Naive Bayes Classifier ----------------------------------------

    # Training Model
    # Amount of Distinct words in training data
    V = sc.broadcast(words_in_training.count())
    # Conditional probabilities
    # ((label, word), cp)
    rdd_train_labword_cp = full_label_wct_rdd.map(lambda x: (x[0], cond_prob(x[1][0], x[1][1])))
    # Conditional probabilities with count 0
    # (label, cp0)
    rdd_train_lab_cp0 = full_label_wct_rdd.map(lambda x: (x[0][0], cond_prob(0, x[1][1]))).distinct()
    # Prior probability
    # (label, pp)
    rdd_train_lab_pp = doc_numb_in_label_rdd.map(lambda x: (x[0], log(x[1]/all_doc_numb)))
    # Training RDDs
    rdd_train = [rdd_train_labword_cp, rdd_train_lab_cp0, rdd_train_lab_pp]

    # Prediction
    # pred_train = predict(rdd_train_data, rdd_train)
    # print('Training Prediction:', pred_train)
    # print('**** training_prediction *********************************')
    pred_test = predict(rdd_test_data, rdd_train)
    # print('Testing Prediction:', pred_test)
    # print('**** testing_prediction **********************************')

    # Accuracy
    # label_train = sc.textFile(training_label).collect()
    # training_acc = cal_accuracy(label_train, pred_train)
    # print('Training Accuracy: %.2f %%' % (training_acc*100))
    # print('**** training_accuracy *********************************')
    # label_test = sc.textFile('csci8360/p1/data/y_test_vsmall.txt').collect()
    # # label_test = sc.textFile('COMP/Practicum/p1/training/y_test_vsmall.txt').collect()
    # testing_acc = cal_accuracy(label_test, pred_test)
    # print('Testing Accuracy: %.2f %%' % (testing_acc*100))
    # print('**** testing_prediction ********************************')

    # Output Files
    # outpath_train = os.path.join(args['output'], 'pred_train_' + args['size'] + '.json')
    # output_file(pred_train, outpath_train)
    outpath_test = os.path.join(args['output'], 'pred_test_' + args['size'] + '.json')
    output_file(pred_test, outpath_test)
