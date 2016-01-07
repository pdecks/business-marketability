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
From completing the engineering exercise in implementing a CSV parser class without the use of Python's built-in csv module, the choice
was clear how I would parse the data ... using the csv module, of course! The following sample shows the header and the first sample from DS_train.csv:

```
unique_id,city,state,contact_title,category,PRMKTS,EQMKTS,RAMKTS,MMKTS,has_facebook,has_twitter,degree_connected,revenue,headcount
VYDAQRLK,New York,NY,Manager,Insurance,0.518403977655587,0.6246364292624702,0.5197196212855738,0.5542533427345436,True,False,2.0,3862000,262
```

When looking at the data, several questions immediately came to mind:

*What do PRMKTS, EQMKTS, RAMKTS, and MMKTS stand for?
*How important is the location?
*How important a predictor are the social media components? (Facebook, Twitter)
*What is the degree connected?
*What value is the revenue in?
*Currency. What if this data isn't all in the same currency? (This would prevent us from normalizing the revenue.)
*Would using ALL of this information overfit the data? What is actually relevant?

###model.py Functions for Preparing Data
loads_data(filepath): To make it easier to access the data later, this function loads the CSV values into a list of dictionaries, one dictionary per each entry (row) where the dictionary keys are the fieldnames from the CSV header. 

list_of_dicts_to_np(list_of_dicts, fields=None): For ease of computations using Scikit Learn, this function converts the list of dictionaries to a single numpy array. The 'fields' parameter allows for the use of a subset of fields. If none, the function uses all values, both numerical and non-numerical.

loads_labels_to_np(filepath, list_of_dicts, id_field): Loads the JSON target labels and matches to the samples in list_of_dicts using the specified 'id_field'. Here, 'id_field'='unique_id'


##Feature Selection
Features are distinct traits that can be used to describe each business in a quantitative manner. In the case of CSV files, it is relatively straightforward to extract features, because the data is structured. An example of unstructured data would be a text document where the number of words varies in each document.

Categorical features, such as the business' location, have no obvious numerical representation but can easily be converted
to a numerical feature. For each distinct location, we can create a new feature that can be valued to 1.0 if the category is matching or 0.0 if not.

```
Example:
Feature 1: "New York, NY"
Feature 2: "Denver, CO"
Feature 3: Temperature, in degrees

Sample 1: "New York, NY", 40 --> [1, 0, 40]
Sample 2: "Denver, CO", 15 --> [0, 1, 15]
```

Questions that arose when initially picking features included:
* How many cities are in the dataset? Would this result in a large number of unimportant features? (spare matrix)
* Should we instead use states? Or maybe clusters of cities (metro areas)? Or maybe city population? (not given here)

## Trial 1:
Using only market values: ['PRMKTS', 'RAMKTS', 'EQMKTS', 'MMKTS']

The advantage of starting with numerical values is that it was straightforward to create a features matrix for all of the samples, given that the LinearSVC model is expecting numerical input. This also seemed like a reasonable place to start, as I assumed, perhaps incorrectly, that the MKTS values were equally important.


##Choose an Algorithm
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

###Model Selection

Following the [Scikit Learn Machine Learning Map](http://scikit-learn.org/stable/tutorial/machine_learning_map/),
because the data has more than 50 samples but less than 100k samples and we have targets for the training data, a LinearSVC, a specific kind of support vector machine, is the most appropriate classifier choice.

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression, and outliers detection. The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is. 

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, the method is likely to perform poorly.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.

For this exercise, we certainly have less than 3,000 features, even if we considered each unique city and unique contact as its own feature.Therefore, we should expect LinearSVC to perform well.

LinearSVC is a specific form of traditional C-Support Vector Classification that uses a linear kernel. Of note is that in Scikit Learn, LinearSVC is implemented in terms of liblinear rather than libsvm, which offers more flexibility in the choice of penalties and loss functions and **should scale better to large numbers of samples**. For traditional SVC, the fit time complexity is more than quadratic with the number of samples, which makes it hard to scale to datasets with more than a few 10,000 samples. Also of note is that the algorithm underlying LinearSVC is [very sensitive to extreme values in its input](http://stackoverflow.com/questions/20624353/why-cant-linearsvc-do-this-simple-classification). 

This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme. Scikit Learn suggests that for optimal performance, one should use a C-ordered numpy.ndarray (dense input).


###Model Hyperparameters
Hyperparameters are model parameters set before the training process. According to Scikit Learn, "parameters that are not directly learnt within estimators can be set by searching a parameter space for the best [cross-validation] score." We can tune hyperparameters using the built-in function [Grid Search](http://scikit-learn.org/stable/modules/grid_search.html#grid-search). 

In the case of [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), grid search takes the following hyperparameters:

```
class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)¶
```

Of these, the following is relevant to this exercise:

*C: Penalty parameter of the error term. A valuation of "how badly" you want to properly fit the data.
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


###Solutions to Overfitting
Evaluating the quality of the model on the data used to fit the model can lead to overfitting. The solution to this issue is twofold:

* Split your data into two sets to detect overfitting situations:
one for training and model selection: the training set
one for evaluation: the test set
* Avoid overfitting by using simpler models (e.g. linear classifiers instead of gaussian kernel SVM) or by increasing the regularization parameter of the model if available (see the docstring of the model for details)
* When the amount of labeled data available is small, it may not be feasible to construct training and test sets. In that case, you can choose to use k-fold cross validation: divide the dataset into k = 10 parts of (roughly) equal size, then for each of these ten parts, train the classifier on the other nine and test on the held-out part

Thus for this exercise, I made use of Scikit Learn's built in helper function cross_val_score:

Using LinearSVC with default hyperparameters and using cross_val_score with 10 folds:
```
>>> print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Accuracy: 0.75 (+/- 0.01)
```

This result is similar to the best score produced by the grid search above.

* Bias and Variance http://www.astroml.org/sklearn_tutorial/practical.html#astro-biasvariance


##Discussion


##Future Work
Using sklearn's DictVectorizer, which is used to convert feature arrays represented as lists of standard Python dict objects to the NumPy/SciPy representation used by scikit-learn estimators, to include location and industry fields. http://scikit-learn.org/stable/modules/feature_extraction.html


###Implementing Chi-Square Test for Feature Importance
One way to evaluate feature importance is using a chi-square test. In statistics, the chi-square test is applied to test the independence of two events. A high chi-square value indicates that the hypothesis of independence is incorrect.

Selecting features using Chi-square aims to simplify the classifier by training on only the "most important" features. A "weaker" classifier -- trained on fewer features -- is often preferable when training data is limited.


##Extra Info

According to Scikit Learn, "[LinearSVC's] underlying C implementation uses a random number generator to select features when fitting the model. It is thus not uncommon to have slightly different results for the same input data. If that happens, try with a smaller tol parameter.

For regression, Y must be a numeric vector with the same number of elements as the number of rows of X.
For classification, Y can be any of these data types.


