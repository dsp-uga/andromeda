# Scalable Document Classification

This repository contains a Naive Bayes classifier implemented on document classification which is completed on CSCI 8360, Data Science Practicum at the University of Georgia, Spring 2018.

This project uses the Reuters Corpus, a set of news stories split into a [hierarchy of categories](), but only specifies four categories as follows:

1. **CCAT**: Corporate / Industrial
2. **ECAT**: Economics
3. **GCAT**: Government / Social
4. **MCAT**: Markets

For those documents with more than one categories (which usually happen), we regard them as if we observed the same document once for each categories. For instance, for the document with CCAT and MCAT categories, we duplicate the document, and pair one of them with CCAT and one of them with MCAT.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Apache Spark 2.2.1](http://spark.apache.org/)
- [Pyspark 2.2.1](https://pypi.python.org/pypi/pyspark/2.2.1) - Python API for Apache Spark
- [Google Cloud Platform](https://cloud.google.com)
- [Anaconda](https://www.anaconda.com/) - packages manager for [nltk](), [string]()

### Installing

### Anaconda

Anaconda is a complete Python distribution embarking automatically the most common packages, and allowing an easy installation of new packages.

Download and install Anaconda (https://www.continuum.io/downloads).

### PyCharm or IntelliJ Idea

IntelliJ Idea is a complete IDE with, between others, Java, Scala and Python pluggins. PyCharm is an equivalent IDE, but with Python as only pluggin (therefore lighter).

Download one of those two IDEs (community edition)
* PyCharm: https://www.jetbrains.com/pycharm/download/
* IntelliJ Idea: https://www.jetbrains.com/idea/download/

If you choose IntelliJ Idea, you must install the Python pluggin, which is not incorporated by default.

### Spark

Download the latest, pre-built for Hadoop 2.6, version of Spark.
* Go to http://spark.apache.org/downloads.html
* Choose a release (prendre la dernière)
* Choose a package type: Pre-built for Hadoop 2.6 and later
* Choose a download type: Direct Download
* Click on the link in Step 4
* Once downloaded, unzip the file and place it in a directory of your choice

## IDE settings to work with PySpark

### Anaconda Interpreter

One first needs to add specific PySpark paths to the ones of the Anaconda Interpreter
* Open your chosen IDE
* Open the cloned Python project with the Anaconda Interpreter
* (IntelliJ only) File -> Project Structure -> SDKs -> your Anaconda interpreter
* (PyCharm only) File -> Default Settings -> Project Interpreter -> your Anaconda interpreter
* (PyCharm only) Click on the "..." icon on the right of your interpreter path, then on "More...", your project interpreter, and finally on the last icon on the bottom right ("Show paths for the selected interpreter")
* Click on "+"
* Select your_path_to_spark/spark-X.X.X-bin-hadoopX.X/python
* "OK"
* Click once again on "+"
* Select your_path_to_spark/spark-X.X.X-bin-hadoopX.X/python/lib/py4j-X.X-src.zip
* Cliquer sur OK
* OK -> Apply -> OK

### Project's environment variables

Finally, we have to set the specific PySpark environment variables to be able to run it in local.

* Run -> Edit Configurations -> Defaults -> Python
* In the "Environment variables" section, click on "...", then on "+"
* Cliquer sur l'icône "+"
* Name: PYTHONPATH
* Value: your_path_to_spark/spark-X.X.X-bin-hadoopX.X/python:your_path_to_spark/spark-X.X.X-bin-hadoopX.X/python/lib/py4j-X.X-src.zip
* Click again on "+"
* Name: SPARK_HOME
* Value: your_path_to_spark/spark-X.X.X-bin-hadoopX.X
* OK -> Apply
* Add the same paths for each test module you will use (Python tests - Unittests for example). Add them for every test module to not have any problem later
* OK

The PySpark imports in your code should now be recognized, and the code should be able to run without any error.

## Running the tests

You can run `p1.py` via regular **python** or run the script via **spark-submit**. You should specify the path to your spark-submit.

```
$ python p1.py [file-directory] [optional args]
```
```
$ usr/bin/spark-submit p1.py [file-directory] [optional args]
```

The output file `pred_test_<size>.json` can be customized by the size you selected, and saved to directory you specified. The required and optional arguments are as follows:

  - **Required Arguments**

    - `path`: Directory contains the input training and testing files

  - **Optional Arguments**

    - `-s`: Sizes to the selected file. (Default: `vsmall`)
    `vsmall` for very small dataset, `small` for small dataset, and `large` for large dataset. The output file will be connected with this selected size, e.g. `pred_test_vsmall.json`.
    - `-o`: Path to the output directory where outputs will be written. (Default: root directory)
    - `-a`: Accuracy of the testing prediction. (Default: `True`)
    The options gives you the accuracy of the prediction. If the file of testing label does not exist, it will still output the file but print out `Accuracy is not available!`.

### Packages Implemented in Preprocessing

After Splitting the document content, we implement punctuation stripping and words stemming by several python APIs. There are some brief explanations about the packages and more details in the  [WIKI](https://github.com/dsp-uga/team-andromeda-p1/wiki) tab.


1. **Punctuation**: `string`

```Python
import string
PUNC = string.punctuation
```

Import the `string` package, then the `string.punctuation` gives you the list of all punctuation.
We remove all punctuation before and after the word, but ignore those punctuation between two words, e.g. `happy--newyear`, `super.....bowl`

2. **Stopwords**: `nltk.corpus`

```Python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
SW = stopwords.words('english')
```

Import the `stopwords` under `nltk.corpus`, then `stopwords.words('english')` gives you the stopwords in English. Notice that you should import `nltk` first, and download `stopwords` from it before importing it from corpus.
We remove those stopwords that might confuse our classifier of the important words, e.g. `the`, `is`, `you`.

3. **Words Stemming**: `nltk.stem`

```Python
import nltk

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
word = lancaster_stemmer.stem(word)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
word = ps.stem(word)

nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()
word = wnl.lemmatize(word)
```

We have tried three stemming packages from `nltk.stem` in this project, the examples of each stemming packages ([Lemmatizer](), [Lancaster](), [Porter]()) are introduced in the [WIKI](https://github.com/dsp-uga/team-andromeda-p1/wiki) tab. Notice that you have to download `wordnet` before importing it.
After implementing words stemming, all the words are transferred to their stems, e.g. `cars`, `car's`, `car` all become `car`.

### Prediction

should say something but i dont know what to say



## Deployment

Add additional notes about how to deploy this on a live system

## Issues

You might encounter different issues when running this classifier on local machine and Google Cloud Platform for the first time. See the following list of issues and solutions or More other issues are documented in the issues Tab: [ISSUES](https://github.com/dsp-uga/team-andromeda-p1/issues)

**Local Machine (Windows)**

- [[Pycharm] Error: too many values to unpack](https://github.com/dsp-uga/team-andromeda-p1/issues/17)
- [[Pyspark] Error: cannot find the file specified](https://github.com/dsp-uga/team-andromeda-p1/issues/18)
- [Worker and Driver has different version](https://github.com/dsp-uga/team-andromeda-p1/issues/19)

**Google Cloud Platform**

- [ImportError: No module named nltk](https://github.com/dsp-uga/team-andromeda-p1/issues/33)
- [ImportError: No module named nltk.stem.wordnet](https://github.com/dsp-uga/team-andromeda-p1/issues/35)
- [Resource wordnet not found](https://github.com/dsp-uga/team-andromeda-p1/issues/36)
- [Name node is in safe mode](https://github.com/dsp-uga/team-andromeda-p1/issues/37)

## Contributing

Please read [CONTRIBUTING.md](https://github.com/dsp-uga/team-andromeda-p1/blob/master/CONTRIBUTORS.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Nihal Soans** - [nihalsoans91](https://github.com/nihalsoans91)
* **Weiwen Xu** - [WeiwenXu21](https://github.com/WeiwenXu21)
* **I-Huei Ho** - [melanieihuei](https://github.com/melanieihuei)

See the [CONTRIBUTOR](https://github.com/dsp-uga/team-andromeda-p1/blob/master/CONTRIBUTORS.md) file for details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Inspiration : Getting an A in Dr Shannon Quinn's Data Science Practicum Course
* Hat tip to anyone who's code was used
* etc
