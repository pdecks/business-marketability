"""Supervised Machine Learning Exercise: Predicting a Business' Marketability"""

"""
Performing NLP using scikit-learn and NLTK.

by Patricia Decker, 11/2015, Hackbright Academy Independent Project

1. Classify Document Category
LinearSVC classifier that takes features vectors consisting of tokenized
reviews that have been converted to numerical values (counts) and
transformed to account for term frequency and inverse document frequency
(tf-idf). Tested on toy data set: 45 hand-labeled reviews that, for the
most part, already contain the word 'gluten'.

2. Perform Sentiment Analysis on Business review
Use NLTK on full-review text to target sentences related to category of
interest and assess sentiment of those target sentences. Generates a
sentiment score for the category based on a probability from 0.0 to 1.0,
where 1.0 is good and 0.0 is bad.
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec

from sklearn.datasets import base as sk_base
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import csv

## DIRECTORIES FOR PICKLING CLASSIFIER & COMPONENTS ##########################

pickle_path_SVC = 'classifiers/LinearSVC/linearSVC.pkl'
pickle_path_v = 'classifiers/LSVCcomponents/vectorizer/linearSVCvectorizer.pkl'
pickle_path_t = 'classifiers/LSVCcomponents/transformer/linearSVCtransformer.pkl'
pickle_path_c = 'classifiers/LSVCcomponents/classifier/linearSVCclassifier.pkl'


## LOAD DATA
def loads_data(filepath):
    """Use builtin CSV module to parse CSV files and return list of dictionaries"""
    all_data = []
    with open(filepath) as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(datareader):
            if i == 0:
                fieldnames = row
            else:
                break
    csvfile.close()

    with open(filepath) as csvfile:     
        datadict = csv.DictReader(csvfile, fieldnames)
        for i, row in enumerate(datadict):
            if i == 0:
                continue
            if 0 < i < 6:
                print
                print "ROW", i
                print row
                all_data.append(row)
            else:
                break
    csvfile.close()

    return all_data, fieldnames


def list_of_dicts_to_np(list_of_dicts, fields=None):
    """Convert list of dictionaries to numpy array for feature extraction.

    fields: list of fieldnames to use. If fields == None, use all fieldnames.
            Else, use subset of fieldnames specified.
    """
    
    # check data types
    if type(list_of_dicts) != list or type(list_of_dicts[0]) != dict:
        raise TypeError('list_of_dicts must be a list of dictionaries')
    
    if fields and type(fields) != list:
        raise TypeError('fields must be a list of field names')

    if not list_of_dicts:
        raise ValueError('list_of_dicts must not be an empty list')

    # set default field subset
    if not fields and list_of_dicts:
        fields = list_of_dicts[0].keys()

    # initialize a np array of zeros
    X = np.zeros([len(list_of_dicts), len(fields)])

    # populate np array with data
    for i, row in enumerate(list_of_dicts):
        for j, field in enumerate(fields):
            X[i, j] = list_of_dicts[i][field]

    return X


train_path = 'data/DS_train.csv'
all_data, fieldnames = loads_data(train_path)
subfields = ['PRMKTS', 'RAMKTS', 'EQMKTS', 'MMKTS']
X = list_of_dicts_to_np(all_data, subfields)

# def create_vectorizer(X_train):
#     """Returns a sklearn vectorizer fit to training data.

#     Input is a numpy array of training data."""
#     # create an instance of CountVectorize feature extractor
#     # using ngram_range flag, enable bigrams in addition to single words
#     count_vect = CountVectorizer(ngram_range=(1, 2))
#     # extract features from training documents' data
#     X_train_counts = count_vect.fit_transform(X_train)

#     return count_vect


# def create_transformer(X_train_counts):
#     """Returns a sklearn transformer fit to training data.

#     Input is a numpy array of training data feature counts."""
#     # create an instance of TfidTransformer that performs both tf & idf
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#     return tfidf_transformer


# ## CREATE AND TRAIN CLASSIFIER ##
# def create_train_classifier(X, y):
#     """Takes documents (X) and targets (y), both np arrays, and returns a trained
#     classifier and its vectorizer and transformer."""

#     X_train = np.copy(X)
#     y_train = np.copy(y)

#     ## EXTRACTING FEATURES ##
#     # TOKENIZATION
#     count_vect = create_vectorizer(X_train)
#     X_train_counts = count_vect.transform(X_train)

#     ## TF-IDF ##
#     tfidf_transformer = create_transformer(X_train_counts)
#     X_train_tfidf = tfidf_transformer.transform(X_train_counts)


#     ## CLASSIFIER ##
#     # Linear SVC, recommended by sklearn machine learning map
#     # clf = Classifier().fit(features_matrix, targets_vector)
#     clf = LinearSVC().fit(X_train_tfidf, y_train)

#     ## CREATING PIPELINES FOR CLASSIFIERS #
#     # Pipeline([(vectorizer), (transformer), (classifier)])
#     pipeline_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf', LinearSVC()),
#                         ])
#     # train the pipeline
#     pipeline_clf = pipeline_clf.fit(X_train, y_train)


#     return (count_vect, tfidf_transformer, clf, pipeline_clf)


# ## SCORE THE CLASSIFIER OVER K-Folds ##
# # add number of features ...

# def score_kfolds(X, y, min_num_folds=2, max_num_folds=2, num_iter=1, atype=None, num_feats=None):
#     """Perform cross-validation on sparse matrix (tf-idf).

#     Returns a dictionary of the scores by fold.
#     atype: if "sentiment", cross-validate sentiment analysis model
#            which assumes the input X is already transformed into a
#            sparse matrix of tf-idf values. if None, assumes X needs
#            to first be vectorized.

#     """
#     if atype is None:
#         count_vect = create_vectorizer(X)
#         X_counts = count_vect.transform(X)

#         tfidf_transformer = create_transformer(X_counts)
#         X_tfidf = tfidf_transformer.transform(X_counts)
#     else:
#         X_tfidf = X

#     if atype == 'stars':
#         clf = LinearSVC()
#     else:
#         clf = LinearSVC()

#     if num_feats:
#         print "Number of features:", num_feats
#         print
#     print "Running score_kfolds with min_num_folds=%d, max_num_folds=%d, num_iter=%d" % (min_num_folds, max_num_folds, num_iter)
#     print "..."
#     start = time.time()
#     # randomly partition data set into 10 folds ignoring the classification variable
#     # b/c we want to see how the classifier performs in this real-world situation

#     # start with k=2, eventually increase to k=10 with larger dataset
#     avg_scores = {}
#     all_avg_scores = {}
#     for k in range(min_num_folds, max_num_folds + 1):
#         avg_scores[k] = {}
#         all_avg_scores[k] = {}
#     for k in range(min_num_folds, max_num_folds + 1):
#         n_fold = k
#         print "Fold number %d ..." % k
#         # run k_fold num_iter number of times at each value of k (2, 3, ..., k)
#         # take average score for each fold, keeping track of scores in dictionary
#         k_dict = {}
#         all_scores = {}
#         for i in range(1, n_fold + 1):
#             k_dict[i] = []
#             all_scores[i] = []
#         #
#         for j in range(1, num_iter +1):
#             k_fold = KFold(n=X_tfidf.shape[0], n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))
#             # print "iteration: %d ..." % j
#             i = 1
#             for train, test in k_fold:
#                 score = clf.fit(X_tfidf[train], y[train]).score(X_tfidf[test], y[test])
#                 y_predict = clf.predict(X_tfidf[test])
#                 accuracy = accuracy_score(y[test], y_predict)
#                 precision = precision_score(y[test], y_predict)
#                 recall = recall_score(y[test], y_predict)
#                 all_scores[i].append((accuracy, precision, recall))
#                 k_dict[i].append(score)
#                 # print "Fold: {} | Score:  {:.4f}".format(i, score)
#                 # k_fold_scores = np.append(k_fold_scores, score)
#                 i += 1
#         #
#         avg_scores[k] = k_dict
#         all_avg_scores[k] = all_scores
#         #
#         print "Iterations for fold %d complete." % k
#     #
#     print '\n-- K-Fold Cross Validation --------'
#     print '-- Mean Scores for {} Iterations --\n'.format(j)
#     for k in range(min_num_folds, max_num_folds + 1):
#       print '-- k = {} --'.format(k)
#       for i in range(1, k+1):
#         print 'Fold: {} | Mean Score: {}'.format(i, np.array(avg_scores[k][i]).mean())
#         if num_iter > 0:
#             print 'Fold: {} | Mean Accuracy Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 0].A1))
#             print 'Fold: {} | Mean Precision Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 1].A1))
#             print 'Fold: {} | Mean Recall Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 2].A1))
#     #
#     endtime = time.time()
#     elapsed = endtime - start
#     print "\nAnalysis completed in", elapsed
#     return (avg_scores, all_avg_scores)


# def tunes_parameters(X, y, n_fold=2):
#     """Perform cross-validation on sparse matrix (tf-idf).

#     Returns a dictionary of the scores by fold."""

#     count_vect = create_vectorizer(X)
#     X_counts = count_vect.transform(X)

#     tfidf_transformer = create_transformer(X_counts)
#     X_tfidf = tfidf_transformer.transform(X_counts)

#     clf = LinearSVC()

#     k_fold = KFold(n=len(X), n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))

#     # pass the entirity of the data, X_tfidf, to cross_val_score
#     # cv is the number of folds for cross-validation
#     # use classification accuracy as deciding metric
#     scores = cross_val_score(clf, X_tfidf, y, cv=10, scoring='accuracy')
#     print scores

#     return scores


# ## PERSIST A COMPONENT OF THE MODEL ##
# def to_persist(items_to_pickle=None, pickling_paths=None):
#     """
#     Takes a list of components to pickle and a list of paths for each item
#     to be pickled.
#     """
#     # todo: check pipeline case...
#     if items_to_pickle and pickling_paths and len(items_to_pickle) == len(pickling_paths):
#         for item, path in zip(items_to_pickle, pickling_paths):

#             decision = raw_input("Would you like to persist %s?\nPath: %s\n(Y) or (N) >>"  % (str(item), str(path)))
#             if decision.lower() == 'y':
#                 persist_component(item, path)
#             else:
#                 print '%s not pickled.' % (str(item))
#                 print

#     print "Persistance complete."
#     return


# def persist_component(component, pickle_path):
#     """Use joblib to pickle the individual classifier components"""
#     joblib.dump(component, pickle_path)
#     print 'Component %s pickled to directory: %s' % (str(component), pickle_path)
#     print
#     return


# ## REVIVE COMPONENT ##
# def revives_component(pickle_path):
#     """Takes the name of the pickled object and returns the revived model.

#     ex: clf_revive = pickle.loads(pdecks_trained_classifier)
#     """
#     component_clone = joblib.load(pickle_path)
#     return component_clone


# ## CLASSIFY NEW REVIEW
# def categorizes_review(review_text, count_vect, tfidf_transformer, clf):
#     """Takes an array containing review text and returns the most relevant
#     category for the review.

#     new_doc_test = ['This restaurant has gluten-free foods.']
#     new_doc_cv = count_vect.transform(new_doc_test)
#     new_doc_tfidf = tfidf_transformer.transform(new_doc_cv)
#     new_doc_category = clf_revive.predict(new_doc_tfidf)
#     print "%s => %s" % (new_doc_test[0], categories[new_doc_category[0]])
#     """

#     # TODO: decide if it is necessary to continually pickle/unpickle every time
#     # the classifier is used

#     # TODO: unpickle classifier
#     # clf_revive = revives_model(pdecks_trained_classifier)

#     text_to_classify = review_text
#     text_to_classify_counts = count_vect.transform(text_to_classify)
#     text_to_classify_tfidf = tfidf_transformer.transform(text_to_classify_counts)
#     new_doc_category = clf.predict(text_to_classify_tfidf)

#     # TODO: pickle classifier
#     # pdecks_trained_classifier = pickles_model(clf_revive)

#     return new_doc_category


# def get_category_name(category_id):
#     """Takes a category index and returns a category name."""
#     return categories[category_id]


# ## FEATURE EXTRACTION ##

# def sorted_features (feature_names, X_numerical, y, kBest=None):
#     """
#     Use chi-square test scores to select top N features from vectorizer.

#     Aims to simplify the classifier by training on only the most important
#     features. The relative importance of the features is important in text
#     classification. Chi-square feature selection can be used to rank features
#     but is not appropriate for making statements about statistical dependence
#     or independence of variables. [see Stanford NLP]

#     feature_names: vectorizer vocabulary, vectorizer.get_feature_names()
#     X: numpy sparse matrix of vectorized documents (can also be tf-idf transformed)
#     y: numpy array of labels (target vector)
#     kBest: integer value of number of best features to extract

#     Returns a list of the features as the words themselves in descending order
#     of importance.
#     """
#     print "\nDetermining best features using chi-square test ..."
#     if not kBest:
#         kBest = X_numerical.shape[1]
#     ch2 = SelectKBest(chi2, kBest)

#     X_numerical = ch2.fit_transform(X_numerical, y)

#     # ch2.get_support() is an array of booleans, where True indicates that
#     # the feature is among the bestK features
#     # ch2.get_support(indicies=True) returns an array of the best feature indices
#     # feature_names[i] maps the index to the vocabulary from the vectorizer to
#     # retrieve the word at that index
#     # best_feature_names is not ranked from best to worst
#     best_feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
#     best_feature_names = np.asarray(best_feature_names)

#     # sort on score in descending order, but provide index and score.
#     top_ranked_features = sorted(enumerate(ch2.scores_),key=lambda x:x[1], reverse=True)[:kBest]

#     # zip(*top_ranked_features) splits the list of kBest (rank, score) tuples into 2 tuples:
#     # 0: kBest-long tuple (best index, ... , least best index)
#     # 1: kBest-long tuple (best score, ... , least best score)
#     # top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
#     top_ranked_features_indices = [x for x in zip(*top_ranked_features)[0]]

#     # ranked from best to worst
#     top_ranked_feature_names = np.asarray([feature_names[i] for i in top_ranked_features_indices])

#     # P-values
#     # for feature_pvalue in zip(np.asarray(train_vectorizer.get_feature_names())[top_ranked_features_indices],ch2.pvalues_[top_ranked_features_indices]):
#     #     print feature_pvalue
#     # # np.asarray(vectorizer.get_feature_names())[ch2.get_support()]
#     print "Feature ranking complete!"
#     return top_ranked_feature_names


# def represents_int(s):
#     """Helper function for checking if input string represents an int"""
#     try:
#         int(s)
#         return True
#     except ValueError:
#         return False


# def get_folds_and_iter():
#     """
#     Prompt the user for number of Kfolds (min and max) and iterations

#     This information is passed to score_kfolds
#     """

#     # cross-validate
#     # minimum K
#     min_num_folds = raw_input("Enter a minimum number of folds (2-10): ")
#     while not represents_int(min_num_folds):
#         min_num_folds = raw_input("Enter a number of folds (2-10): ")
#         print
#     min_num_folds = int(min_num_folds)

#     # maximum K
#     if int(min_num_folds) != 10:
#         max_num_folds = raw_input("Enter a maximum number of folds (%d-10): " % int(min_num_folds))
#         while not represents_int(max_num_folds) or int(max_num_folds) < min_num_folds:
#             max_num_folds = raw_input("Enter a maximum number of folds (%d-10): " % int(min_num_folds))
#             print
#     else:
#         max_num_folds = 10

#     # number of iterations
#     num_iter = raw_input("Enter a number of iterations (1-50): ")
#     print

#     while not represents_int(num_iter):
#         num_iter = raw_input("Enter a number of iterations (1-50): ")
#         print


#     max_num_folds = int(max_num_folds)
#     num_iter = int(num_iter)

#     return (min_num_folds, max_num_folds, num_iter)


# def train_classifier():
#     """SUPERCEDED by multilabel classifier
#     Trains the classifier on the labeled yelp data.

#     Tests the classifier pipeline on a "new doc".

#     Provides opportunities to persist the trained model and/or its components
#     """
#     # LOAD the training documents
#     # documents = loads_pdecks_reviews(container_path, categories)
#     documents = loads_yelp_reviews(container_path, categories)
#     X, y = bunch_to_np(documents)

#     min_num_folds, max_num_folds, num_iter = get_folds_and_iter()
#     fold_avg_scores = score_kfolds(X, y, min_num_folds, max_num_folds, num_iter)

#     scores = tunes_parameters(X, y, 10)

#     # CREATE and TRAIN the classifier PIPELINE
#     X, y = bunch_to_np(documents)
#     count_vect, tfidf_transformer, clf, pipeline_clf = create_train_classifier(X, y)

#     # TEST the classifier
#     new_doc = ['I love gluten-free foods. This restaurant is the best.']
#     new_doc_category_id_pipeline = pipeline_clf.predict(new_doc)
#     new_doc_category_pipeline = get_category_name(new_doc_category_id_pipeline)

#     print
#     print "-- Test document --"
#     print
#     print "Using Pipeline:"
#     print "%r => %s" % (new_doc[0], new_doc_category_pipeline)

#     # PERSIST THE MODEL / COMPONENTS
#     items_to_pickle = [pipeline_clf, count_vect, tfidf_transformer, clf]
#     pickling_paths = [pickle_path_SVC, pickle_path_v, pickle_path_t, pickle_path_c]
#     to_persist(items_to_pickle=items_to_pickle, pickling_paths=pickling_paths)

#     return





# if __name__ == "__main__":

#     ## TRAIN AND PERSIST CLASSIFIER 
#     print '\n--SUPERCEDED CLASSIFIERS--'
#     to_train = raw_input("Train the LinearSVC classifier for categorization? Y or N >> ")
#     if to_train.lower() == 'y':
#         train_classifier()

#     ## CHECK PERFORMANCE OF PICKLED CLASSIFIER ON DATA SET
#     else:
#         print
#         to_test = raw_input("Check the pipeline classifier on the toy data set? Y or N >>")
#         if to_test.lower() == 'y':
#             check_toy_dataset()
