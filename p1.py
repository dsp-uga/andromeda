import argparse
import json
import os.path
import numpy as np
import string
from nltk.corpus import stopwords
from collections import Counter
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from operator import add

from pyspark import SparkContext


def book_to_terms(book):
  """
  Converts a book to a list of individual words
  """
  _, contents = book
  # contents.split() will generate a bunch of individual tokens. Each term (word)
  # in this list is then run through a *local* map that strips any remaining
  # whitespace off either side of the word and converts it to lowercase.
  words = list(map(lambda word: word.strip().lower(), contents.split()))
  return words

def terms_to_counts(term):
  """
  Converts each term to a tuple with a count of 1.
  """
  return (term, 1)

def combine_by_word(count1, count2):
  """
  This simply adds two word counts (we don't know what the key is; we just
  know that it's the same for both counts, which is why we're adding them).

  This will also work as-is to add up NumPy arrays of counts for subproject D.
  """
  return count1 + count2

def count_threshold(word_count):
  """
  Drops any word counts less than 2.
  """
  word, count = word_count
  return count > 2

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



def doc2vec(doc_tuple): #<- <docid> <content> <label>
  """
  This takes the same document tuple that is the output of wholeTextFiles,
  and parses out all the words for a single document, AND builds the
  document-specific count vectors for each word.
  """
  docid, content, label = doc_tuple

  # This is how we know what document we're in--i.e., what document
  # count to increment in the count array.
  document_list = DOCS.value
  doc_index = document_list.index(docid)

  # Generate a list of words and do a bunch of processing.
  words = book_to_terms(["junk", content])

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
    count_vector = np.zeros(N, dtype = np.int)
    count_vector[doc_index] += 1

    # Build a list of (word, vector) tuples. I'm returning them all at
    # one time at the very end, but you could just as easily make use
    # of the "yield" keyword here instead to return them one-at-a-time.
    out_tuples.append([w, count_vector]) #<- [<word> [count in each doc]]
  return out_tuples

def remove_punctuation_from_end(word):
  punctuation = PUNC.value
  if len(word)>0 and word[0] in punctuation:
    word = word[1:]
  if len(word)>0 and word[-1] in punctuation:
    word = word[:-1]
  return word

def check_punctuation(word):
  punctuation = PUNC.value
  while len(word)>0 and (word[0] in punctuation or word[-1] in punctuation):
    word = remove_punctuation_from_end(word)
  return word

# def remove_punctuation_advanced(word):
#     '''
#     Replace double-hyphen to white space, and result in two separate words.
#     Remove the punctuations before or after the string.
#     '''
#     if '--' in word:
#         words = word.replace('--', ' ')
#     translator = str.maketrans('', '', PUNC)
#     if "n't" not in word:
#         strip = list(Counter(word.translate(translator)).split())
#         word = strip[0]
#     return word


def counts_to_tfidf(word_vector):
  """
  Computes the TF-IDF scores for each term-vector pair.
  """
  word, vector = word_vector
  N = vector.shape[0]
  nt = vector[vector > 0].shape[0] # Number of documents in which word appears.

  # Compute IDF and TF-IDF.
  idf = np.log(N / nt)
  tfidf = np.array([tf * idf for tf in vector])
  return (word, tfidf)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "CSCI 8360 Project 1",
		epilog = "answer key", add_help = "How to use",
		prog = "python p1.py [train-data] [train-label] [test-data] [optional args]")

	# Required args
	parser.add_argument("paths", required = True, nargs=3,
		help = "Paths of training-data, training-labels, and testing-data.")

  # Optional args
# 	parser.add_argument("-s", "--stopwords", default = None,
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

  # Preprocessing
  rdd = rdd.map(lambda x: (x[0], x[1].split(',')))
  rdd = rdd.flatMapValues(lambda x: x)\
		.filter(lambda x: 'CAT' in x[1]) #<content> <label_containing_'CAT'>
  rdd = rdd.zipWithIndex().map(lambda x: (x[1], x[0][0], x[0][1])) # <doc_id> <content> <label>

  doc_numb = rdd.count()
  DOCS = sc.broadcast(range(doc_numb))
	
  frequency_vectors = rdd.map(doc2vec)	
	
	
	
	
# 	keep going
	

