#Understanding the Problem 
This is a supervised learning exercise because we are supplied with target labels for the training dataset. Supervised learning can be further broken down into two categories, classification and regression. In classification, the label is discrete, while in regression, the label is continuous. Here, we have discrete labels that also happen to be binary values. Therefore, this exercise is a binary classification problem. 


##Questions that arose when looking at the headers for the training data
```
unique_id,city,state,contact_title,category,PRMKTS,EQMKTS,RAMKTS,MMKTS,has_facebook,has_twitter,degree_connected,revenue,headcount
VYDAQRLK,New York,NY,Manager,Insurance,0.518403977655587,0.6246364292624702,0.5197196212855738,0.5542533427345436,True,False,2.0,3862000,262
```


What do PRMKTS, EQMKTS, RAMKTS, and MMKTS stand for?
How important is the location?
How important a predictor are the social media components? (Facebook, Twitter)
What is the degree connected?
What value is the revenue in? Can look at relative values and disregard specific
currency. What if this data isn't all in the same currency?
Would using ALL of this information overfit the data? What is actually relevant?

#Machine Learning Model
Often the hardest part of solving a machine learning problem can be finding the right estimator for the job, as different estimators are better suited for different types of data and different problems.

The steps for supervised learning are:

* Prepare Data
* Choose an Algorithm
* Fit a Model
* Choose a Validation Method
* Examine Fit and Update Until Satisfied
* Use Fitted Model for Predictions

For regression, Y must be a numeric vector with the same number of elements as the number of rows of X.
For classification, Y can be any of these data types. This table also contains the method of including missing entries.

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

##Model Selection

Following the (#Scikit Learn Machine Learning Map)http://scikit-learn.org/stable/tutorial/machine_learning_map/,
the data has more than 50 samples but less than 100k samples and we have target labels for the training data.
According to this guide, a LinearSVC, a kind of support vector machine, is the most appropriate classifier choice.

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, the method is likely to give poor performances.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

For this exercise, we have at most 12 features (one per column heading) and 3,000 samples, therefore we should expect this method to perform well.

Scikit Learn suggests that for optimal performance, one should use a C-ordered numpy.ndarray (dense).

LinearSVC is similar to traditional C-Support Vector Classification (SVC) as it uses a linear kernel. In sklearn, LinearSVC is implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and **should scale better to large numbers of samples**. For traditional SVC, the fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a few 10000 samples.

This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.

###Model Hyperparameters
Hyperparameters are model parameters set before the training process. "The underlying C implementation uses a random number generator to select features when fitting the model. It is thus not uncommon to have slightly different results for the same input data. If that happens, try with a smaller tol parameter.""

###Solutions to Overfitting
Evaluating the quality of the model on the data used to fit the model can lead to overfitting. The solution to this issue is twofold:

* Split your data into two sets to detect overfitting situations:
one for training and model selection: the training set
one for evaluation: the test set
* Avoid overfitting by using simpler models (e.g. linear classifiers instead of gaussian kernel SVM) or by increasing the regularization parameter of the model if available (see the docstring of the model for details)
* When the amount of labeled data available is small, it may not be feasible to construct training and test sets. In that case, you can choose to use k-fold cross validation: divide the dataset into k = 10 parts of (roughly) equal size, then for each of these ten parts, train the classifier on the other nine and test on the held-out part

####sklearn.cross_validation.cross_val_score
Built in helper function

Using LinearSVC with default hyperparameters and using cross_val_score with 10 folds:
```
>>> print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Accuracy: 0.75 (+/- 0.01)
```

* Bias and Variance http://www.astroml.org/sklearn_tutorial/practical.html#astro-biasvariance

#Feature Selection
Features are distinct traits that can be used to describe each business in a quantitative manner. In the case of CSV files, it is relatively straightforward to extract features, because the data is structured. An example of unstructured data would be a text document where the number of words varies
Categorical features, such as the business' location, have no obvious numerical representation but can easily be converted
to a numerical feature. For each distinct location, we can create a new feature that can be valued to 1.0 if the category is matching or 0.0 if not.

Example:
Feature 1: "New York, NY"
Feature 2: "Denver, CO"

How many cities are in the dataset?
Should we just use states? Or maybe clusters of cities (metro areas)? By city population?

## Trial 1:
Using only market values: ['PRMKTS', 'RAMKTS', 'EQMKTS', 'MMKTS']

The advantage of starting with numerical values is that it was straightforward to create a features matrix for all of the samples, given that the LinearSVC model is expecting numerical input.

##Chi-Square Test

#Analysis


#Future Studies
Using sklearn's DictVectorizer, which is used to convert feature arrays represented as lists of standard Python dict objects to the NumPy/SciPy representation used by scikit-learn estimators, to include location and industry fields. http://scikit-learn.org/stable/modules/feature_extraction.html


