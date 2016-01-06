# Understanding the Problem 
```
unique_id,city,state,contact_title,category,PRMKTS,EQMKTS,RAMKTS,MMKTS,has_facebook,has_twitter,degree_connected,revenue,headcount
VYDAQRLK,New York,NY,Manager,Insurance,0.518403977655587,0.6246364292624702,0.5197196212855738,0.5542533427345436,True,False,2.0,3862000,262
```

### Questions that arose when looking at the headers for the training data

What do PRMKTS, EQMKTS, RAMKTS, and MMKTS stand for?
How important is the location?
How important a predictor are the social media components? (Facebook, Twitter)
What is the degree connected?
What value is the revenue in? Can look at relative values and disregard specific
currency. What if this data isn't all in the same currency?
Would using ALL of this information overfit the data? What is actually relevant?

# Machine Learning Model
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

Because the target labels are binary values, this exercise is a binary classification problem. Some of the methods commonly used for binary classification are:

* Decision trees
* Random forests
* Bayesian networks
* Support vector machines
* Neural networks
* Logistic regression

Following the (#Scikit Learn Machine Learning Map)http://scikit-learn.org/stable/tutorial/machine_learning_map/,
the data has more than 50 samples but less than 100k samples and we have target labels for the training data.
According to this guide, a LinearSVC, a kind of support vector machine, is the most appropriate classifier choice.

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, the method is likely to give poor performances.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

For this exercise, we have at most 12 features (one per column heading) and 3,000 samples, therefore we should expect this method to perform well.

# Feature Selection


# Analysis



