<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Team-Andromeda

# Project 1: Scalable Document Classification

For this project, we are using the Reuters Corpus, which is a set of news stories split into
a hierarchy of categories. There are multiple class labels per document, but for the sake
of simplicity we’ll ignore all but the labels ending in CAT:
1. CCAT: Corporate / Industrial
2. ECAT: Economics
3. GCAT: Government / Social
4. MCAT: Markets
There are some documents with more than one CAT label. Treat those documents as if
you observed the same document once for each CAT label. For example, if a document
has both the labels CCAT and MCAT, you will essentially duplicate that document and give
one of them only the label CCAT, and the other only the label MCAT.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- Spark
- Python
- GCP or a Awesome Computer at home
- Hadhoop

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

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Issues when Running for the First Time

The Issue are documented in the issues Tab

[ISSUES](https://github.com/dsp-uga/team-andromeda-p1/issues)


## Contributing

Please read [CONTRIBUTING.md](https://github.com/dsp-uga/team-andromeda-p1/blob/master/CONTRIBUTORS.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Nihal** - [nihalsoans91](https://github.com/nihalsoans91)
* **Jenny** - [WeiwenXu21](https://github.com/WeiwenXu21)
* **Melanie** - [melanieihuei](https://github.com/melanieihuei)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the UnLicense - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Inspiration : Getting an A in Dr Shannon Quinn's Data Science Practicum Course
* Hat tip to anyone who's code was used
* etc
