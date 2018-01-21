import argparse
import json
import os.path
import numpy as np
import string
from operator import add

from pyspark import SparkContext

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

	
