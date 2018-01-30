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
    """
    This tokenizes individual words
    """
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

def cleanup_word(word):
    """
    This completes all the preprocessing required.
    Including removing punctuation and steming words
    """
    w = ''.join([i for i in word if not i.isdigit()])
    w = check_punctuation(w)
    # lancaster_stemmer = LancasterStemmer()
    # w = lancaster_stemmer.stem(w)
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    w = ps.stem(wnl.lemmatize(w.lower()))
    w = check_punctuation(w)
    return w

def doc2vec(doc_tuple): #<- <label> <content>
    """
    This takes the same document tuple that is the output of wholeTextFiles,
    and parses out all the words for a single document, AND builds the
    document-specific count vectors for each word.

    input: doc_tuple
           <content>

    output: <word>, [c1,c2,c3,...,c_n]
    """
    content, label = doc_tuple
    # This is how we know what document we're in--i.e., what document
    # count to increment in the count array.
#     label_list = LABELS.value
#     document_list = DOCS.value
#     doc_index = document_list.index(docid)

    # Generate a list of words and do a bunch of processing.
    no_quot_words = content.replace("--", " ").split("&quot")
    words = tokenize_words(no_quot_words)
    # words = book_to_terms(["junk", content])

    out_tuples = []
#    N = len(document_list) # Denominator for TF-IDF.
#     punctuation = PUNC.value
    stopwords = SW.value
    for w in words:
        # Enforce stopwords and minimum length.
        w = cleanup_word(w)
        if w in stopwords or len(w) <= 1: continue
# #         w = check_punctuation(w)
# #         lancaster_stemmer = LancasterStemmer()
# #         w = lancaster_stemmer.stem(w)
#         # Build the document-count vector.
#         count_vector = []
#         for i in range(N):
#             count_vector.append([i, label_list[i], 0])
#         # initialize [[<docid(0)>, <label>, <count(0)>],
#         #             [<docid(1)>, <label>, <count(0)>],
#         #             [<docid(2)>, <label>, <count(0)>],...]
#         count_vector[doc_index][2] += 1

#     # Build a list of (word, vector) tuples. I'm returning them all at
#     # one time at the very end, but you could just as easily make use
#     # of the "yield" keyword here instead to return them one-at-a-time.
        out_tuples.append(w)
    # [<word> [[<docid(0)>, <label(0)>, <count>],
    #          [<docid(1)>, <label(1)>, <count>],
    #          [<docid(2)>, <label(0)>, <count>],...]]
    return (label, out_tuples)

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

# def laplace_smoothing(word_count_in_label, count_in_label):
#     """
#     This applys laplace smoothing to obtain the conditional probability of
#     a word given specific label c. v is the total number of possible words in training set.
#     """
#     v = V.value
#     nominator = word_count_in_label + 1
#     denominator = count_in_label + v
#     cond_prob = log(nominator / denominator)
#     return cond_prob
#
# def laplace_cond_prob(word_list):
#     """
#     This counts the conditional probability of each word, given the specific label,
#     the probabilities are applied laplace smoothing.
#     """
#     total_count = 0
#     for i in range(len(word_list)):
#         total_count += word_list[i][1]
#     for i in range(len(word_list)):
#         word_list[i] = (word_list[i][0], laplace_smoothing(word_list[i][1], total_count))
#         # word_list[i] = (word_list[i][0], word_list[i][1] / total_count)
#     return word_list

def cond_prob_rdd(rdd):
    v = V.value
    total_counts = sum(rdd.map(lambda x: x[1]).collect())
    cond_prob_rdd = rdd.map(lambda x: (x[0], log((x[1]+1)/(total_counts+v))))
    return cond_prob_rdd

# def cond_prob_rdd(cp_rdd, rdd):
#     """
#     This changes former rdd with word counts into rdd with word conditional probabilities.
#     The conditional probabilities here have already been applied laplace smoothing.
#     """
#     labels = rdd.map(lambda x: x[0]).distinct().collect()
#
#     for ind in range(len(labels)):
#         rdd_same_label = rdd.filter(lambda x: x[0]==labels[ind])
#         rdd_label = rdd_same_label.map(lambda x: x[0]).distinct()
#
#         list_same_label = rdd_same_label.flatMap(lambda x: tuple(x[1])).reduceByKey(add).collect()
#         sum_count_in_label = rdd_label.map(lambda x: (x, list_same_label))
#         rdd_cond_prob = sum_count_in_label.map(lambda x: (x[0], laplace_cond_prob(x[1])))
#         cp_rdd = cp_rdd.union(rdd_cond_prob)
#     return cp_rdd
#
# def prior_prob_rdd(pp_rdd, rdd):
#     """
#     This changes former rdd grouped by document ids
#     into rdd grouped by different labels.
#     """
#     doc_num = rdd.count()
#     labels = rdd.map(lambda x: x[0]).distinct().collect()
#
#     for ind in range(len(labels)):
#         rdd_same_label = rdd.filter(lambda x: x[0]==labels[ind])
#         rdd_label = rdd_same_label.map(lambda x: x[0]).distinct()
#
#         doc_num_same_label = rdd_same_label.count()
#         rdd_prior_prob = rdd_label.map(lambda x: (x, log(doc_num_same_label/doc_num)))
#         pp_rdd = pp_rdd.union(rdd_prior_prob)
#     return pp_rdd

# in rdd
# def NBtraining(cp_rdd, pp_rdd):
#     """
#     This returns a huge rdd combined by cp_rdd and pp_rdd,
#     and transforms each probability value.
#     """
#     # input: <label>, [(w1, cp1), (w2, cp2), ..., (wd,cpd)], pp
#     # output: <label>, [(w1, cp1), (w2, cp2), ..., (wd,cpd), pp]
#     rdd = cp_rdd.leftOuterJoin(pp_rdd)\
#                 .map(lambda x: (x[0], [x[1][0], x[1][1]]))
#     return rdd

# def words_list(word_n_count_list):
#     """
#     This transfers list of word and its counts to a list that contains only words in each file.
#     Each line of the rdd will be the word list of each document.
#     """
#     length = len(word_n_count_list)
#     new_list = []
#
#     for i in range(length):
#         if not (word_n_count_list[i][1] == 0):
#             new_list.append(word_n_count_list[i][0])
#
#     # [["w_1", "w_2", "w_3", ......, "w_d1"],
#     #  ["w_1", "w_2", "w_3", ......, "w_d2"], ...,
#     #  ["w_1", "w_2", "w_3", ......, "w_dk"]]
#     return new_list

def docSpec_vec(content):
    no_quot_words = content.replace("--", " ").split("&quot")
    words = tokenize_words(no_quot_words)
    stopwords = SW.value
    test_word_list = []
    for w in words:
        w = cleanup_word(w)
        if w in stopwords or len(w) <= 1: continue
        test_word_list.append([w,1])
    return test_word_list

def validation_format(data_rdd_in_textfile):
# a function that transfers inputting sc.textFile() into the format as
# rdd([doc_1, [['w11', 1], ['w12', 1], ..... , ['w1d', 1]]],
#     [doc_2, [['w21', 1], ['w22', 1], ..... , ['w2d', 1]]], ...)
    docid_data_rdd = data_rdd_in_textfile.zipWithIndex()
    tokenized_docid_data_rdd = docid_data_rdd.map(lambda x: (x[1],docSpec_vec(x[0])))
    return tokenized_docid_data_rdd
    # return docid_data_rdd

def calculate(values, label):
    if (values[1][1] != None):
        return values[1][0] * values[1][1] #>>> depends on how you set validation format
    else:
        return label[2]

def NBpredict(cp_rdd_list, val_testing_rdd):
    """
    This provides a list of prediction of labels for each document in testing data.
    """
    add = ADD.value
    doc_numb = val_testing_rdd.count()
    prediction = []

    for i in range(doc_numb):
        # testing_doc = testing_rdd.map(lambda x: x[1] if (x[0]==i) else continue) <<
        test_doc = val_testing_rdd.filter(lambda x: x[0]==i)
        testing_doc = test_doc.map(lambda x:x[1]).flatMap(lambda x: x[:])
        prob = []
        for label in add:
            training_cp = cp_rdd_list[add.index(label)]
            new_rdd = testing_doc.leftOuterJoin(training_cp)
            cal_rdd = new_rdd.map(lambda x: calculate(x, label))
            # train_label = training_rdd.filter(lambda x: x[0]==label[0])
            # training_label_cp = train_label.map(lambda x: x[1][0]).flatMap(lambda x: x[:])
            # new_rdd = testing_doc.leftOuterJoin(training_label_cp)
            # cal_rdd = new_rdd.map(lambda x: calculate(x,label))

            log_p = sum(cal_rdd.collect()) + label[1]
            prob.append(log_p)

        max_index = np.argmax(prob)
        prediction.append(add[max_index][0])

    return prediction

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
    rdd_train_data = sc.textFile(training_data).zipWithIndex().map(lambda x: (x[1],x[0]))
    rdd_train_label = sc.textFile(training_label).zipWithIndex().map(lambda x: (x[1],x[0]))
    rdd_test_data = sc.textFile(testing_data).zipWithIndex().map(lambda x: (x[1],x[0]))
#     rdd = rdd_train_data.zip(rdd_train_label)
    rdd = rdd_train_data.join(rdd_train_label).map(lambda x: x[1])

    # Preprocessing --------------------------------------------------
    # Preprocessing to labels, leaving only ones with 'CAT'
    rdd = rdd.map(lambda x: (x[0], x[1].split(',')))
    rdd = rdd.flatMapValues(lambda x: x).filter(lambda x: 'CAT' in x[1])
    # Document Numbers for each label
    doc_numb_in_label_rdd = rdd.map(lambda x: (x[1],1)).reduceByKey(lambda x,y: x+y)
    doc_numb_in_label = doc_numb_in_label_rdd.collect()


    # Precessing to content of each document
#    rdd_train_data = rdd.map(lambda x: x[0]).zipWithIndex()

#    DOCS = sc.broadcast(rdd_train_data.map(lambda x:x[1]).collect())
    #
    # Assuming we have n docs
    rdd = rdd.map(doc2vec)
    rdd = rdd.groupByKey().map(lambda x : (x[0], list(x[1])))

    counts = []
    total_doc_numb = 0
    total_count_in_each_label = []
    for i in doc_numb_in_label:
        total_doc_numb += i[1]
        label = i[0]
        rdd_label = rdd.filter(lambda x: x[0] == label).flatMap(lambda x: x[1]).flatMap(lambda x: x)
        rdd_label_count = rdd_label.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
        rdd_label_count = rdd_label_count.filter(lambda x: x[1]>1)
        counts.append(rdd_label_count)
        total_word_count = sum(rdd_label_count.map(lambda x: x[1]).collect())
        l = []
        l.append(label)
        l.append(total_word_count)
        total_count_in_each_label.append(l)

    # Total Counts for each label
    total_count_in_each_label_rdd = sc.parallelize(total_count_in_each_label)
    # Distinct Words List in training data
    words_in_training = rdd.flatMap(lambda x: x[1]).flatMap(lambda x: x)\
                            .distinct().map(lambda x: (x,0))
    V = sc.broadcast(words_in_training.count())

    counts_full = []
    for rdd_item in counts:
        new_rdd = rdd_item.union(words_in_training).reduceByKey(lambda x,y: x+y)
        counts_full.append(new_rdd)

    # Naive Bayes Classifier ----------------------------------------

    # Conditional Probabilities, P(word|label)
    # (after laplace smoothing and log transformation)
    cp_rdd_list = []
    for rdd in counts_full:
        cp_rdd_list.append(cond_prob_rdd(rdd))
    # Prior Probability + Conditional Probability with count 0
    # (after laplace smoothing and log transformation)
    v = V.value
    pp_rdd = doc_numb_in_label_rdd.map(lambda x: [x[0], log(x[1]/total_doc_numb)])
    cp0_rdd = total_count_in_each_label_rdd.map(lambda x: [x[0], log(1/x[1]+v)])
    ADD = pp_rdd.leftOuterJoin(cp0_rdd)\
                .map(lambda x: [x[0], x[1][0], x[1][1]])
    ADD = sc.broadcast(ADD.collect())

    # Prediction
    val_training_rdd = validation_format(rdd_train_data)
    val_testing_rdd = validation_format(rdd_test_data)
    prediction_train = NBpredict(cp_rdd_list, val_training_rdd)
    print('Training Prediction:', prediction_train)
    prediction_test = NBpredict(cp_rdd_list, val_testing_rdd)
    print('Testing Prediction:', prediction_test)

    # Accuracy
    # 1, training accuracy
    label_train = rdd_train_label.collect()
    training_acc = cal_accuracy(label_train, prediction_train)
    print('Training Accuracy: %.2f %%' % (training_acc*100))
    # 2, testing accuracy
    rdd_test_label = sc.textFile('~/csci8360/p1/data/y_test_vsmall.txt')
    label_test = rdd_test_label.collect()
    testing_acc = cal_accuracy(rdd_test_label, prediction_test)
    print('Testing Accuracy: %.2f %%' % (testing_acc*100))

    # Output Files
    outF = open("pred_test.json", "w")
    textList = '\n'.join(prediction_test)
    outF.writelines(textList)
    outF.close()


#     doc_spec_frequency_vectors, with_id = wordSpec2docSpec(word_specific_frequency_vectors)
#     #rdd[[<label0>,[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]],
#     #    ...
#     #    [<labelN>,[[<'word0'>,<count>],[<'word1'>,<count>],...,[<'wordN'>,<count>]]]

#     #word list in each document (??)
#     words_in_doc_rdd = with_id.map(lambda x: (x[0][0],words_list(x[1])))
# #     words_in_doc_rdd_nodocid = words_in_doc_rdd.map(lambda x: x[1])

#     words_in_label_rdd = doc_spec_frequency_vectors.map(lambda x: (x[0],words_list(x[1])))
#     word_count_rdd = words_in_label_rdd.reduceByKey(lambda x,y: x+y)
#     #word count in each label
#     word_count_each_label_rdd = word_count_rdd.map(lambda x: (x[0],len(set(x[1]))))

#     # Naive Bayes classifier
#     word_numb = word_specific_frequency_vectors.count()
#     #number of distinct words in training set
#     V = sc.broadcast(word_numb)
#     #broadcast number of distinct words in training set

#     # model training
#     cp_rdd = sc.parallelize([])
#     cp_rdd = cond_prob_rdd(cp_rdd, doc_spec_frequency_vectors)
#     pp_rdd = sc.parallelize([])
#     pp_rdd = prior_prob_rdd(pp_rdd, doc_spec_frequency_vectors)
#     NB_training_rdd = NBtraining(cp_rdd, pp_rdd)

#     # model testing
#     ADD = pp_rdd.leftOuterJoin(word_count_each_label_rdd)\
#                 .map(lambda x: (x[0], x[1][0], laplace_smoothing(0, x[1][1])))
#     ADD = sc.broadcast(ADD.collect())
#     val_training_rdd = validation_format(rdd_train_data)
#     val_testing_rdd = validation_format(rdd_test_data)
#     # prediction
#     prediction_train = NBpredict(NB_training_rdd, val_training_rdd)
#     prediction_test = NBpredict(NB_training_rdd, val_testing_rdd)
#     print('Training Prediction:', prediction_train)
#     print('Testing Prediction:', prediction_test)
#     # accuracy
#     # 1, training accuracy
#     label_train = rdd_train_label.collect()
#     training_acc = cal_accuracy(label_train, prediction_train)
#     print('Training Accuracy: %.2f %%' % (training_acc*100))
#     # 2, testing accuracy
#     # testing_acc = cal_accuracy(rdd_test_label, prediction_test)
#     # print('Testing Accuracy: %.2f %%' % (testing_acc*100))

#     # Output files
#     outF = open("pred_test.json", "w")
#     textList = '\n'.join(prediction_test)
#     outF.writelines(textList)
#     outF.close()
