"""Supervised Machine Learning Exercise: Predicting a Business' Marketability

by Patricia Decker, 1/6/2016

Objectives:
1. Featurize training data
2. Train a model
3. Use the model to predict business marketability

"""

import random
import math
import numpy as np
import csv
import json
import time

from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.externals import joblib
from sklearn.preprocessing import scale


## DIRECTORY FOR PICKLING CLASSIFIER 
pickle_path_SVC = 'classifiers/LinearSVC/linearSVC.pkl'

## LOAD DATA
def loads_data(filepath):
    """Use builtin CSV module to parse CSV files and return list of dictionaries"""
    all_data = []
    with open(filepath) as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(datareader):
            if i == 0:
                fieldnames = row
            elif i == 1:
                # fieldtypes = [type(x) for x in row]  # return all <type 'str'>
                fieldtypes = []
                for s in row:
                    if represents_int(s):
                        fieldtypes.append(int)
                    elif represents_float(s):
                        fieldtypes.append(float)
                    elif represents_bool(s) != -1:
                        fieldtypes.append(bool)
                    else:
                        fieldtypes.append(str)
            else:
                break
    csvfile.close()

    with open(filepath) as csvfile:     
        datadict = csv.DictReader(csvfile, fieldnames)
        for i, row in enumerate(datadict):
            if i == 0:
                continue
            all_data.append(row)
            # if 0 < i < 100:
                # print
                # print "ROW", i
                # print row
                # all_data.append(row)
            # else:
                # break
    csvfile.close()

    return all_data, fieldnames, fieldtypes


def list_of_dicts_to_np(list_of_dicts, fieldnames=None, fieldtypes=None, dictvect=False):
    """Convert list of dictionaries to numpy array for feature extraction.

    fields: list of fieldnames to use. If fields == None, use all fieldnames.
            Else, use subset of fieldnames specified.
    dictvect: flag for using DictVectorizer, which transforms lists of feature-
            value mappings to vectors. Returns np array

    """
    
    # check data types
    if type(list_of_dicts) != list or type(list_of_dicts[0]) != dict:
        raise TypeError('list_of_dicts must be a list of dictionaries')
    
    if fieldnames and type(fieldnames) != list:
        raise TypeError('fieldnames must be a list of strings')

    if fieldtypes and type(fieldtypes) != list:
        raise TypeError('fieldtypes must be a list of strings')

    if not list_of_dicts:
        raise ValueError('list_of_dicts must not be an empty list')

    # set default field subset and determine field types
    if not fieldnames and not fieldtypes and list_of_dicts:
        fieldnames = list_of_dicts[0].keys()
        # fieldtypes = [type(x) for x in row]
        fieldtypes = []
        for s in fieldnames:
            if represents_int(s):
                fieldtypes.append(int)
            elif represents_float(s):
                fieldtypes.append(float)
            elif represents_bool(s) != -1:
                fieldtypes.append(bool)
            else:
                fieldtypes.append(str)

    # Found that DictVectorizer resulted in a sparse matrix of shape (3000, 17709)
    # if dictvect == True:
    #     dv = DictVectorizer(sparse=False)
    #     # todo: update to only use desired fields
    #     X = dv.fit_transform(list_of_dicts)

    # TODO: determine size of matrix to create if not all types are numerical or boolean
    if str in fieldtypes:
        pass 
    else:  # contains int, float, or bool only
        # initialize a np array of zeros
        X = np.zeros([len(list_of_dicts), len(fieldnames)])

        # populate np array with data
        for i, row in enumerate(list_of_dicts):
            for j, field in enumerate(fieldnames):
                if fieldtypes[j] == bool:
                    if represents_bool(list_of_dicts[i][field]):
                        X[i,j] = 1
                    else:
                        X[i,j] = 0
                elif fieldtypes[j] == int:
                    X[i, j] = int(list_of_dicts[i][field])
                else:
                    X[i, j] = float(list_of_dicts[i][field])

    
    return X


def loads_labels_to_np(filepath, list_of_dicts, id_field):
    """Load JSON target labels and match to list_of_dicts samples"""

    # check data types
    if type(list_of_dicts) != list or type(list_of_dicts[0]) != dict:
        raise TypeError('list_of_dicts must be a list of dictionaries')
    
    if not list_of_dicts:
        raise ValueError('list_of_dicts must not be an empty list')


    with open(filepath) as data_file:    
        json_data = json.load(data_file)

    y = [0] * len(list_of_dicts)
    
    for i, row in enumerate(list_of_dicts):
        y[i] = json_data[row[id_field]]

    return np.array(y)


## SCORE THE CLASSIFIER OVER K-Folds ##
# add number of features ...

def score_kfolds(X, y, min_num_folds=2, max_num_folds=2, num_iter=1, num_feats=None):
    """Performs cross-validation and returns a dictionary of the scores by fold.

    X should be a dense matrix, otherwise it needs to be transformed.
    """
    clf = LinearSVC()

    if num_feats:
        print "Number of features:", num_feats
        print
    print "Running score_kfolds with min_num_folds=%d, max_num_folds=%d, num_iter=%d" % (min_num_folds, max_num_folds, num_iter)
    print "..."
    start = time.time()
    # randomly partition data set into 10 folds ignoring the classification variable
    # b/c we want to see how the classifier performs in this real-world situation

    # start with k=2, eventually increase to k=10 with larger dataset
    avg_scores = {}
    all_avg_scores = {}
    for k in range(min_num_folds, max_num_folds + 1):
        avg_scores[k] = {}
        all_avg_scores[k] = {}
    for k in range(min_num_folds, max_num_folds + 1):
        n_fold = k
        print "Fold number %d ..." % k
        # run k_fold num_iter number of times at each value of k (2, 3, ..., k)
        # take average score for each fold, keeping track of scores in dictionary
        k_dict = {}
        all_scores = {}
        for i in range(1, n_fold + 1):
            k_dict[i] = []
            all_scores[i] = []
        #
        for j in range(1, num_iter +1):
            k_fold = KFold(n=X.shape[0], n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))
            # print "iteration: %d ..." % j
            i = 1
            for train, test in k_fold:
                score = clf.fit(X[train], y[train]).score(X[test], y[test])
                y_predict = clf.predict(X[test])
                accuracy = accuracy_score(y[test], y_predict)
                precision = precision_score(y[test], y_predict)
                recall = recall_score(y[test], y_predict)
                all_scores[i].append((accuracy, precision, recall))
                k_dict[i].append(score)
                # print "Fold: {} | Score:  {:.4f}".format(i, score)
                # k_fold_scores = np.append(k_fold_scores, score)
                i += 1
        #
        avg_scores[k] = k_dict
        all_avg_scores[k] = all_scores
        #
        print "Iterations for fold %d complete." % k
    #
    print '\n-- K-Fold Cross Validation --------'
    print '-- Mean Scores for {} Iterations --\n'.format(j)
    for k in range(min_num_folds, max_num_folds + 1):
      print '-- k = {} --'.format(k)
      for i in range(1, k+1):
        print 'Fold: {} | Mean Score: {}'.format(i, np.array(avg_scores[k][i]).mean())
        if num_iter > 0:
            print 'Fold: {} | Mean Accuracy Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 0].A1))
            print 'Fold: {} | Mean Precision Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 1].A1))
            print 'Fold: {} | Mean Recall Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 2].A1))
    #
    endtime = time.time()
    elapsed = endtime - start
    print "\nAnalysis completed in", elapsed
    return (avg_scores, all_avg_scores)


## PERSIST A COMPONENT OF THE MODEL ##
def to_persist(items_to_pickle=None, pickling_paths=None):
    """
    Takes a list of components to pickle and a list of paths for each item
    to be pickled.
    """
    import pdb; pdb.set_trace()
    if items_to_pickle and pickling_paths and len(items_to_pickle) == len(pickling_paths):
        for item, path in zip(items_to_pickle, pickling_paths):

            decision = raw_input("Would you like to persist %s?\nPath: %s\n(Y) or (N) >>"  % (str(item), str(path)))
            if decision.lower() == 'y':
                persist_component(item, path)
            else:
                print '%s not pickled.' % (str(item))
                print

    print "Persistance complete."
    return

## REVIVE COMPONENT ##
def revives_component(pickle_path):
    """Takes the name of the pickled object and returns the revived model.

    ex: clf_revive = pickle.loads(pdecks_trained_classifier)
    """
    component_clone = joblib.load(pickle_path)
    return component_clone


## TODO: FEATURE EXTRACTION IMPROVEMENT USING CHI-SQUARE ##
## The following code was from my Hackbright project but was not updated for
## this application
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


def represents_int(s):
    """Helper function for checking if input string represents an int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def represents_float(s):
    """Helper function for checking if input string represents a float"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def represents_bool(s):
    """Helper function for checking if input string represents a boolean"""
    if s == 'True' or s == 'true' or s == 'T':
        return True
    elif s == 'False' or s == 'false' or s == 'F':
        return False
    else:
        return -1


def get_folds_and_iter():
    """
    Prompt the user for number of Kfolds (min and max) and iterations

    This information is passed to score_kfolds
    """

    # cross-validate
    # minimum K
    min_num_folds = raw_input("Enter a minimum number of folds (2-10): ")
    while not represents_int(min_num_folds):
        min_num_folds = raw_input("Enter a number of folds (2-10): ")
        print
    min_num_folds = int(min_num_folds)

    # maximum K
    if int(min_num_folds) != 10:
        max_num_folds = raw_input("Enter a maximum number of folds (%d-10): " % int(min_num_folds))
        while not represents_int(max_num_folds) or int(max_num_folds) < min_num_folds:
            max_num_folds = raw_input("Enter a maximum number of folds (%d-10): " % int(min_num_folds))
            print
    else:
        max_num_folds = 10

    # number of iterations
    num_iter = raw_input("Enter a number of iterations (1-50): ")
    print

    while not represents_int(num_iter):
        num_iter = raw_input("Enter a number of iterations (1-50): ")
        print


    max_num_folds = int(max_num_folds)
    num_iter = int(num_iter)

    return (min_num_folds, max_num_folds, num_iter)


def train_classifier():
    """Trains the classifier on the labeled data.

    Provides opportunities to persist the trained model and/or its components
    """

    # define filepaths
    test_path = 'data/DS_test.csv'
    train_path = 'data/DS_train.csv'
    json_path = 'data/DS_train_labels.json'

    # load the training data
    all_data, fieldnames, fieldtypes = loads_data(train_path)

    # create np arrays for training data and target labels
    
    # Trial 1: Numerical Data Only
    # subfields = ['PRMKTS', 'RAMKTS', 'EQMKTS', 'MMKTS']
    # subtypes = [fieldtypes[fieldnames.index(x)] for x in subfields]
    # X = list_of_dicts_to_np(all_data, subfields )
    
    # Trial 2a: use all fields, use dictvectorize
    # X = list_of_dicts_to_np(all_data, dictvect=True)
    
    # Trial 2b: use subfields, numerical data and booleans only (no dictvectorize)
    subfields = ['PRMKTS', 'RAMKTS', 'EQMKTS', 'MMKTS', 'has_facebook', 'has_twitter']
    subtypes = [fieldtypes[fieldnames.index(x)] for x in subfields]

    # normalize the data before fitting
    # X = scale(X)  # this had a neglible effect on the accuracy

    X = list_of_dicts_to_np(all_data, subfields, subtypes)
    y = loads_labels_to_np(json_path, all_data, 'unique_id')

    count = 0
    for val in y:
        if val == 1:
            count += 1
    print "\nNumber of 1s in targets:", count
    print "Number of 0s in targets:", len(y) - count

    decision = raw_input("\nPerform manual cross-validation? Y or N >> ")
    if decision.lower() == 'y':
        min_num_folds, max_num_folds, num_iter = get_folds_and_iter()
        fold_avg_scores = score_kfolds(X, y, min_num_folds, max_num_folds, num_iter)

    clf = LinearSVC()
    
    # CROSS VALIDATION
    print "\nPerforming Cross Validation with 10 folds ... "
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    # Print mean score and the 95% confidence interval of the score estimate 
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # GRID SEARCH
    print "\nPerforming Grid Search to tune hyperparameters ..."
    params_space = {'C': np.logspace(-5, 0, 10), 'class_weight':[None, 'auto']}
    gscv = GridSearchCV(clf, params_space, cv=10, n_jobs=-1)
    # train the model
    gscv.fit(X, y)
    # give a look at your best params combination and best score you have
    print "Best Estimator:"
    print gscv.best_estimator_
    print "Best Parameters:"
    print gscv.best_params_
    print "Best Score:"
    print gscv.best_score_

    # CREATE and TRAIN the classifier
    clf = LinearSVC(C=0.278).fit(X, y)

    # PREDICT targets for test data
    decision = raw_input("\nPredict targets for test data and create JSON file? Y or N >> ")
    if decision.lower() == 'y':
        # load the training data
        test_data, fieldnames, fieldtypes = loads_data(test_path)

        X_test = list_of_dicts_to_np(test_data, subfields, subtypes)

        y_test = clf.predict(X_test)

        results = {}
        for i, sample in enumerate(test_data):
            results[sample['unique_id']] = y_test[i]

        with open('data/test_predictions.json', 'w') as outfile:
            json.dump(results, outfile)
                    
    # PERSIST THE MODEL / COMPONENTS
    # items_to_pickle = [clf]
    # pickling_paths = [pickle_path_SVC]
    
    # to_persist(items_to_pickle=items_to_pickle, pickling_paths=pickling_paths)

    return


if __name__ == "__main__":

    ## TRAIN AND PERSIST CLASSIFIER 
    to_train = raw_input("Train the LinearSVC classifier for categorization? Y or N >> ")
    if to_train.lower() == 'y':
        train_classifier()
