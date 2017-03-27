## Movie Reviews - Sentiment Analysis
`Python 3.5` classification of tweets (positive or negative) using `NLTK-3` and `sklearn`.

An analysis of the `twitter` data set included in the `nltk` corpus.

***
## What is in this repo


- [x] An implementation of `nltk.NaiveBayesClassifier` trained against **1000 tweets**. Implemented in `Train_Classifiers.py`.
- [x] Using `sklearn`
  - [x] **Naive Bayes**: 
    - [x] `MultinomialNB`: 
    - [x] `BernoulliNB`:
  - [x] **Linear Model**
    - [x] `LogisticRegression`:
    - [x] `SGDClassifier`:
  - [x] **SVM**
    - [x] `SVC`: 
    - [x] `LinearSVC`:
    - [x] `NuSVC`:

Implemented in `Scikit_Learn_Classifiers.py`

- [x] Implemented a voting system to choose the best out of all the learning methods. Implemented in `sentiment_mod.py`

***

### Accuracy achieved


| **Classifiers**                 | **Accuracy achieved** |
|---------------------------------|-----------------------|
| `nltk.NaiveBayesClassifier`     | _73.0%_               |
| **ScikitLearn Implementations** |                       |
| `BernoulliNB`                   | _72.0%_               |
| `MultinomialNB`                 | _75.0%_               |
| `LogisticRegression`            | _71.0%_               |
| `SGDClassifier`                 | _69.0%_               |
| `SVC`                           | _48.0%_               |
| `LinearSVC`                     | _74.0%_               |
| `NuSVC`                         | _75.0%_               |

***

## Requirements


The simplest way(and the suggested way) would be to install the the required packages and the dependencies by using either [anaconda](https://www.continuum.io/downloads) or [miniconda](http://conda.pydata.org/miniconda.html)

After that you can do

```sh
$ conda update conda
$ conda install scikit-learn nltk
```

***

#### Downloading the dataset


The dataset used in this package is bundled along with the `nltk` package.

Run your python interpreter

```python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('movie_reviews') 
```

**NOTE**: You can check system specific installation instructions from the official [`nltk` website](http://www.nltk.org/data.html)

Check if everything is good till now by running your interpreter again and importing these

```python
>>> import nltk
>>> from nltk.corpus import stopwords, movie_reviews
>>> import sklearn
>>> 
```

If these imports work for you. Then you are good to go!

***

## Running it

1. Clone the repo 

```sh
$ git clone https://github.com/aalind0/Movie_Reviews-Sentiment_Analysis
$ cd Movie_Reviews-Sentiment_Analysis
```

2. Order of running
  1. `NLTK_Naive_Bayes.py`
  2. `Scikit_Learn_Classifiers.py`
  3. `Voting_Algos.py`

3. Hack away!

***

## So

**"So what, Well this is pretty basic!"**

Yes, it is but hey we all do start somewhere right?

**Coming Up**. I am working on a Twitter Sentiment Analysis project which first trains on a given data-set and then takes in the live twitter feeds, analyses them plus plots them for data visualization.

You can follow me on twitter [@singh_aalind](https://twitter.com/singh_aalind) to keep tabs on it. 

***

## End

Hacked together by [Aalind Singh](https://aalind0.github.io).


