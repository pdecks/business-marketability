#Machine Learning for Business Marketability
##Understanding the Problem 
This is a supervised learning exercise because we are supplied with targets for the training dataset. Supervised learning can be further broken down into two categories, classification and regression. In classification, the label is discrete, while in regression, the label is continuous. Here, we have discrete labels that also happen to be binary values. Therefore, this exercise is a binary classification problem. 

The steps for supervised learning are:

* Prepare Data
* Choose an Algorithm
* Fit a Model
* Choose a Validation Method
* Examine Fit and Update Until Satisfied
* Use Fitted Model for Predictions

##Prepare Data
From completing the engineering exercise in implementing a CSV parser class without the use of Python's built-in csv module, the choice was clear how I would parse the data ... using the csv module, of course! The following sample shows the header and the first sample from DS_train.csv:

```
unique_id,city,state,contact_title,category,PRMKTS,EQMKTS,RAMKTS,MMKTS,has_facebook,has_twitter,degree_connected,revenue,headcount
VYDAQRLK,New York,NY,Manager,Insurance,0.518403977655587,0.6246364292624702,0.5197196212855738,0.5542533427345436,True,False,2.0,3862000,262
```

###Balanced Data
Imbalanced data typically refers to a problem with classification problems where the classes are not represented equally. Looking at the target values, we find the following:

```
Number of 1s in targets: 817
Number of 0s in targets: 2183
```

Therefore, of the 3,000 samples in the training data, about 25% of them are in class 1 (target = 1) and 75% are in class 2 (target = 2), such that the ratio of instances in class 1 to instances in class 2 is 1:3, which is imbalanced.

###model.py Functions for Preparing Data
**loads_data(filepath)**: To make it easier to access the data later, this function loads the CSV values into a list of dictionaries, one dictionary per each entry (row) where the dictionary keys are the fieldnames from the CSV header. 

**list_of_dicts_to_np(list_of_dicts, fieldnames=None, fieldtypes=None, dictvect=False)**: For ease of computations using scikit-learn (sklearn), this function converts the list of dictionaries to a single numpy array. The 'fieldnames' parameter allows for the use of a subset of fields input as a list, for which 'fieldtypes' is a corresponding list of types for each field. If none, the function uses all values, both numerical and non-numerical. 'dictvect' is a flag for using sklearn's built-in DictVectorizer, which transforms lists of feature-value mappings to vectors.

**loads_labels_to_np(filepath, list_of_dicts, id_field)**: Loads the JSON target labels and matches to the samples in list_of_dicts using the specified 'id_field'. Here, 'id_field'='unique_id'


##Feature Selection
Features are distinct traits that can be used to describe each business in a quantitative manner. In the case of CSV files, it is relatively straightforward to extract features, because the data is structured. An example of unstructured data would be a text document where the number of words varies in each document.

Categorical features, such as the business' location, have no obvious numerical representation but can easily be converted to a numerical feature. For each distinct location, we can create a new feature that can be valued to 1.0 if the category is matching or 0.0 if not.

```
Example:
Feature 1: "New York, NY"
Feature 2: "Denver, CO"
Feature 3: Temperature, in degrees

Sample 1: "New York, NY", 40 --> [1, 0, 40]
Sample 2: "Denver, CO", 15 --> [0, 1, 15]
```

When looking at the data, several questions immediately came to mind related to feature selection:

* What do PRMKTS, EQMKTS, RAMKTS, and MMKTS stand for?
* How important is the location?
* How many cities are in the dataset? Would this result in a large number of unimportant features? (spare matrix)
* Should we instead use states? Or maybe clusters of cities (metro areas)? Or maybe city population? (not given here)
* How important a predictor are the social media components? (Facebook, Twitter)
* What is the degree connected?
* What value is the revenue in?
* What if this data isn't all in the same currency? (This would prevent us from normalizing the revenue.)
* Would using ALL of this information overfit the data? What is actually relevant?

### Trial 1: ['PRMKTS', 'RAMKTS', 'EQMKTS', 'MMKTS']

The advantage of starting with numerical values is that it was straightforward to create a features matrix for all of the samples, given that the LinearSVC model is expecting numerical input. This also seemed like a reasonable place to start, as I assumed, perhaps incorrectly, that the MKTS values were equally important.

### Trial 2: Trial 1 + 'has_facebook' + 'has_twitter'

When evaluating the model accuracy below, it seemed that using only the four fields from Trial 1 was not capturing enough information to accurately predict business marketability. The next logical step seemed to be to include some social media aspects, as social media is often a key component of marketing. But does this depend on the kind of business you are in? If you have a social media account as a manufacturing company that probably does not make as much of an impact as for a dating service that has social media accounts, given that the latter is inherently a more social kind of business.

To make use of the non-binarized information (e.g., location, category, contact_title), I used sklearn's [DictVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html), which converts feature arrays represented as lists of standard Python dict objects to the NumPy/SciPy representation used by sklearn estimators, to include location and industry fields. To become more familiar with DictVectorizer, I used it on the entirity of X, that is, I did not use the intended subfields but instead used all fieldnames. Surprisingly, this resulted in an extremely large and therefore extremely sparse matrix:

```
>>> X
array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ..., 
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]])
>>> X.shape
(3000, 17709)
```

So it seemed that it **is** possible in this case to end up with **n_features > n_samples**, which is important to keep in mind when selecting a model, as is discussed in the Model Selection section. Backing off, I decided to manually convert the 'has_facebook' and 'has_twitter' boolean values to their numerical equivalents, 0 (False) and 1 (True). At this point I went back to model.py's loads_data() to grab the data type information from the first entry, hoping that all of the boolean values are well-formed (i.e., 'False' not 'false'), and then added some type checks to list_of_dicts_to_np() to convert the data.

At this point I ran into a big bug, realizing that all of my data was "<type 'str'>". I thought that I had been working with numerical data, so I had a bit more cleaning to do. The first row of the CSV file as an entry in list_of_dicts is shown below, along with the type for one of the MKTS fields:

```
>>> list_of_dicts[0]
{'category': 'Insurance', 'city': 'New York', 'has_twitter': 'False', 'PRMKTS': '0.518403977655587', 'RAMKTS': '0.5197196212855738', 'EQMKTS': '0.6246364292624702', 'revenue': '3862000', 'state': 'NY', 'has_facebook': 'True', 'MMKTS': '0.5542533427345436', 'degree_connected': '2.0', 'contact_title': 'Manager', 'headcount': '262', 'unique_id': 'VYDAQRLK'}
>>> type(list_of_dicts[0]['PRMKTS'])
<type 'str'>
```

After creating some helper functions to check whether the strings represented int, float, or bool types (**represents_ int**, **represents_ float**, and **represents_bool**, respectively), I re-ran Trial 1 and noted that the results were the same as for when the data consisted of strings. 

##Model Selection
Often the hardest part of solving a machine learning problem can be finding the right estimator for the job, as different estimators are better suited for different types of data and different problems.

There are tradeoffs between several characteristics of algorithms, such as:

* Speed of training
* Memory usage
* Predictive accuracy on new data
* Transparency or interpretability, meaning how easily you can understand the reasons an algorithm makes its predictions

Some of the methods commonly used for binary classification are:

* Decision trees
* Random forests
* Bayesian networks
* Support vector machines
* Neural networks
* Logistic regression

Following the [sklearn Machine Learning Map](http://scikit-learn.org/stable/tutorial/machine_learning_map/),
because the data has more than 50 samples but less than 100k samples and we have targets for the training data, a LinearSVC, a specific kind of support vector machine, is the most appropriate classifier choice.

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression, and outliers detection. The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is. 

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, **n_features > n_samples**, the method is likely to perform poorly.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.

For this exercise, I assumed that we would certainly have less than 3,000 features, even if we considered each unique city and unique contact as its own feature and that we should therefore expect LinearSVC to perform well. However, after using **DictVectorize** I found that a very large, very sparse matrix could result, violating this assumption.

LinearSVC is a specific form of traditional C-Support Vector Classification that uses a linear kernel. Of note is that in sklearn, LinearSVC is implemented in terms of liblinear rather than libsvm, which offers more flexibility in the choice of penalties and loss functions and **should scale better to large numbers of samples**. For traditional SVC, the fit time complexity is more than quadratic with the number of samples, which makes it hard to scale to datasets with more than a few 10,000 samples. Also of note is that the algorithm underlying LinearSVC is [very sensitive to extreme values in its input](http://stackoverflow.com/questions/20624353/why-cant-linearsvc-do-this-simple-classification). 

This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme. sklearn suggests that for optimal performance, one should use a C-ordered numpy.ndarray (dense input).


###Model Hyperparameters
Hyperparameters are model parameters set before the training process. According to sklearn, "parameters that are not directly learnt within estimators can be set by searching a parameter space for the best [cross-validation] score." We can tune hyperparameters using the built-in function [Grid Search](http://scikit-learn.org/stable/modules/grid_search.html#grid-search). 

In the case of [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), grid search takes the following hyperparameters:

```
class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)¶
```

Of these, the following is relevant to this exercise:

* C: Penalty parameter of the error term. A valuation of "how badly" you want to properly fit the data.
[Source: Stack Exchange](http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel)

####Trial 1: Using Only PRMKTS, EQMKTS, RAMKTS, and MMKTS
The results of the grid search for the LinearSVC model for Trial 1 are presented below:

```
Best Estimator:
LinearSVC(C=0.27825594022071259, class_weight=None, dual=True,
     fit_intercept=True, intercept_scaling=1, loss='squared_hinge',
     max_iter=1000, multi_class='ovr', penalty='l2', random_state=None,
     tol=0.0001, verbose=0)

Best Parameters:
{'C': 0.27825594022071259, 'class_weight': None}

Best Score:
0.745666666667
```

This accuracy score is essentially unchanged from the LinearSVC model with C=1.0. It looks like this model does not perform very well for this data. Alternatively, let's see if we can do better by using more features. Perhaps we are missing some of the key indicators of performance by limiting ourselves to the four "MKTS" values.


####Trial 2: Trial 1 + has_facebook + has_twitter
The results of the grid search for the LinearSVC model for Trial 2 are presented below:

```
Best Estimator:
LinearSVC(C=0.27825594022071259, class_weight=None, dual=True,
     fit_intercept=True, intercept_scaling=1, loss='squared_hinge',
     max_iter=1000, multi_class='ovr', penalty='l2', random_state=None,
     tol=0.0001, verbose=0)

Best Parameters:
{'C': 0.27825594022071259, 'class_weight': None}

Best Score:
0.746666666667
```

Clearly, there was neglible change in the scores such that the use of these additional features did not improve performance. The low value for the penalty (C) may indicate that the grid search accounted for the imbalanced dataset. The penalty introduces bias to the model, forcing it to pay more attention to the minority class.

###Solutions to Overfitting
Evaluating the quality of the model on the data used to fit the model can lead to overfitting. The solution to this issue is twofold:

* Split your data into two sets to detect overfitting situations:
one for training and model selection: the training set
one for evaluation: the test set
* Avoid overfitting by using simpler models (e.g. linear classifiers instead of gaussian kernel SVM) or by increasing the regularization parameter of the model if available (see the docstring of the model for details)
* When the amount of labeled data available is small, it may not be feasible to construct training and test sets. In that case, you can choose to use k-fold cross validation: divide the dataset into k = 10 parts of (roughly) equal size, then for each of these ten parts, train the classifier on the other nine and test on the held-out part

Thus for this exercise, I made use of sklearn's built in helper function cross_val_score:

Using LinearSVC with default hyperparameters and using cross_val_score with 10 folds:
```
>>> print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Accuracy: 0.75 (+/- 0.01)
```

This result is similar to the best score produced by the grid search above.

In addition to accuracy, we should also consider precision and recall, which can easily be done using sklearn's built in metrics: accuracy_score, precision_score, and recall_score. The following are the scores for a 10-fold cross-validation using 10 iterations:

```
-- k = 10 --
Fold: 1 | Mean Score: 0.750666666667
Fold: 1 | Mean Accuracy Score: 0.750666666667
Fold: 1 | Mean Precision Score: 0.735594718319
Fold: 1 | Mean Recall Score: 0.1173459856
Fold: 2 | Mean Score: 0.739333333333
Fold: 2 | Mean Accuracy Score: 0.739333333333
Fold: 2 | Mean Precision Score: 0.62837995338
Fold: 2 | Mean Recall Score: 0.0848157995233
Fold: 3 | Mean Score: 0.775
Fold: 3 | Mean Accuracy Score: 0.775
Fold: 3 | Mean Precision Score: 0.76807408278
Fold: 3 | Mean Recall Score: 0.11676876975
Fold: 4 | Mean Score: 0.744666666667
Fold: 4 | Mean Accuracy Score: 0.744666666667
Fold: 4 | Mean Precision Score: 0.738177257525
Fold: 4 | Mean Recall Score: 0.137281430934
Fold: 5 | Mean Score: 0.755333333333
Fold: 5 | Mean Accuracy Score: 0.755333333333
Fold: 5 | Mean Precision Score: 0.626998491704
Fold: 5 | Mean Recall Score: 0.10802702354
Fold: 6 | Mean Score: 0.740333333333
Fold: 6 | Mean Accuracy Score: 0.740333333333
Fold: 6 | Mean Precision Score: 0.737537046287
Fold: 6 | Mean Recall Score: 0.108718739381
Fold: 7 | Mean Score: 0.738666666667
Fold: 7 | Mean Accuracy Score: 0.738666666667
Fold: 7 | Mean Precision Score: 0.737918470418
Fold: 7 | Mean Recall Score: 0.113045829193
Fold: 8 | Mean Score: 0.751333333333
Fold: 8 | Mean Accuracy Score: 0.751333333333
Fold: 8 | Mean Precision Score: 0.760164981218
Fold: 8 | Mean Recall Score: 0.132664668407
Fold: 9 | Mean Score: 0.730666666667
Fold: 9 | Mean Accuracy Score: 0.730666666667
Fold: 9 | Mean Precision Score: 0.693796595561
Fold: 9 | Mean Recall Score: 0.102969325421
Fold: 10 | Mean Score: 0.734
Fold: 10 | Mean Accuracy Score: 0.734
Fold: 10 | Mean Precision Score: 0.77813043166
Fold: 10 | Mean Recall Score: 0.104836387741
```

While the accuracy and precision are relatively good at more than 70%, the recall is very poor, at about 10%. The recall is a measure of a classifier's completeness. According to sklearn, "A system with high recall but low precision returns many results, but most of its predicted labels are incorrect when compared to the training labels. A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. An ideal system with high precision and high recall will return many results, with all results labeled correctly.""

##Future Work: "If I had another day ..."
* Rebalance the training dataset such that there are an equal number of targets in each class by resampling the dataset.
 Over-sampling the data, repeating data points from the underrepresented class, may be a good approach given that we have less than 10,000 samples.
* Implement the chi-square test for feature importance. In statistics, the chi-square test is applied to test the independence of two events. A high chi-square value indicates that the hypothesis of independence is incorrect. Selecting features using Chi-square aims to simplify the classifier by training on only the "most important" features. A "weaker" classifier -- trained on fewer features -- is often preferable when training data is limited. This could therefore improve the classifier's predictions by allowing us to use only the most important features.
* Try a completely different model, or even try SVC using kernel='linear' and compare results. Decision trees such as Random Forest often perform well on imbalanced datasets.
* Make use of string information without creating a severely sparse matrix, perhaps by looking for clusters of locations or finding a way to normalize the cities by population.
* Circle back to DictVectorizer to see if the fact that my dictionary items were all strings (not int, float, bool, and str) was why the sparse matrix had so many features.
* Explore why are trials 1 and 2 produce nearly the same accuracy results


