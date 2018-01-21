import argparse
import json
import os.path
import numpy as np
import string
from operator import add

from pyspark import SparkContext

def book_to_terms(book):
    """
        Converts a book to a list of individual words.
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "CSCI 8360 Project 1",
		epilog = "answer key", add_help = "How to use",
		prog = "python p1.py <options> [optional args]")

	# Required args
	parser.add_argument("-p", "--path", required = True,
		help = "Path to the directory containing all input text files")

	# Optional args
	parser.add_argument("-s", "--stopwords", default = None,
	        help = "Path to a file containing stopwords. [DEFAULT: None]")
	parser.add_argument("-a", "--algorithm", choices = ["NB", "LR"], default = "NB",
		help = "Algorithms to process classification: \"NB\": Naive Bayes, \"LR\": Logistic Regression [Default: Naive Bayes]")
    	parser.add_argument("-o", "--output", default = ".",
        help = "Path to the output directory where outputs will be written. [Default: \".\"]")

	args = vars(parser.parse_args())
    	sc = SparkContext()

	# Read in the variables
    	inputs = args['path']
	algorithm = args['algorithm']

    	# Necessary Lists
    	stopwords = args['stopwords']
    	punctuation = sc.broadcast(string.punctuation)
	
	# Generate an RDD of tuples
    	rdd = sc.wholeTextFile(inputs)

	# Preprocessing
	if algorithm == "NB" or algorithm == "LR":
		terms = rdd.flatMap(book_to_terms)
        	frequencies = terms.map(terms_to_counts) \
			.reduceByKey(combine_by_word)
        	top_frequencies = frequencies.filter(count_threshold) \
			.persist()

	
