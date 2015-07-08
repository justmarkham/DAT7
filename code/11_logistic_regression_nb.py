# # Logistic Regression

# ## Agenda
# 1. Refresh your memory on how to do linear regression in scikit-learn
# 2. Attempt to use linear regression for classification
# 3. Show you why logistic regression is a better alternative for classification
# 4. Brief overview of probability, odds, e, log, and log-odds
# 5. Explain the form of logistic regression
# 6. Explain how to interpret logistic regression coefficients
# 7. Compare logistic regression with other models


# ## Part 1: Predicting a Continuous Response

# glass identification dataset
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
glass = pd.read_csv(url, names=col_names, index_col='id')
glass['assorted'] = glass.glass_type.map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})
glass.head()

# Pretend that we want to predict **ri**, and our only feature is **al**. How would we do it using machine learning? We would frame it as a regression problem, and use a linear regression model with **al** as the only feature and **ri** as the response.

# How would we **visualize** this model? Create a scatter plot with **al** on the x-axis and **ri** on the y-axis, and draw the line of best fit.
import seaborn as sns
import matplotlib.pyplot as plt
sns.lmplot(x='al', y='ri', data=glass, ci=None)

# If we had an **al** value of 2, what would we predict for **ri**? Roughly 1.517.

# **Exercise:** Draw this plot without using Seaborn.

# scatter plot using Pandas
glass.plot(kind='scatter', x='al', y='ri')

# scatter plot using Matplotlib
plt.scatter(glass.al, glass.ri)

# fit a linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['al']
X = glass[feature_cols]
y = glass.ri
linreg.fit(X, y)

# look at the coefficients to get the equation for the line, but then how do you plot the line?
print linreg.intercept_
print linreg.coef_

# you could make predictions for arbitrary points, and then plot a line connecting them
print linreg.predict(1)
print linreg.predict(2)
print linreg.predict(3)

# or you could make predictions for all values of X, and then plot those predictions connected by a line
ri_pred = linreg.predict(X)
plt.plot(glass.al, ri_pred, color='red')

# put the plots together
plt.scatter(glass.al, glass.ri)
plt.plot(glass.al, ri_pred, color='red')


# ### Refresher: interpreting linear regression coefficients

# Linear regression equation: $y = \beta_0 + \beta_1x$

# compute prediction for al=2 using the equation
linreg.intercept_ + linreg.coef_ * 2

# compute prediction for al=2 using the predict method
linreg.predict(2)

# examine coefficient for al
zip(feature_cols, linreg.coef_)

# **Interpretation:** A 1 unit increase in 'al' is associated with a 0.0025 unit decrease in 'ri'.

# increasing al by 1 (so that al=3) decreases ri by 0.0025
1.51699012 - 0.0024776063874696243

# compute prediction for al=3 using the predict method
linreg.predict(3)


# ## Part 2: Predicting a Categorical Response

# Let's change our task, so that we're predicting **assorted** using **al**. Let's visualize the relationship to figure out how to do this:
plt.scatter(glass.al, glass.assorted)

# Let's draw a **regression line**, like we did before:

# fit a linear regression model and store the predictions
feature_cols = ['al']
X = glass[feature_cols]
y = glass.assorted
linreg.fit(X, y)
assorted_pred = linreg.predict(X)

# scatter plot that includes the regression line
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred, color='red')

# If **al=3**, what class do we predict for assorted? **1**
# If **al=1.5**, what class do we predict for assorted? **0**

# So, we predict the 0 class for **lower** values of al, and the 1 class for **higher** values of al. What's our cutoff value? Around **al=2**, because that's where the linear regression line crosses the midpoint between predicting class 0 and class 1.

# So, we'll say that if **assorted_pred >= 0.5**, we predict a class of **1**, else we predict a class of **0**.

# understanding np.where
import numpy as np
nums = np.array([5, 15, 8])

# np.where returns the first value if the condition is True, and the second value if the condition is False
np.where(nums > 10, 'big', 'small')

# examine the predictions
assorted_pred[:10]

# transform predictions to 1 or 0
assorted_pred_class = np.where(assorted_pred >= 0.5, 1, 0)
assorted_pred_class

# plot the class predictions
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_class, color='red')

# What went wrong? This is a line plot, and it connects points in the order they are found. Let's sort the DataFrame by "al" to fix this:

# add predicted class to DataFrame
glass['assorted_pred_class'] = assorted_pred_class

# sort DataFrame by al
glass.sort('al', inplace=True)

# plot the class predictions again
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, glass.assorted_pred_class, color='red')


# ## Part 3: Using Logistic Regression Instead

# Logistic regression can do what we just did:

# fit a linear regression model and store the class predictions
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['al']
X = glass[feature_cols]
y = glass.assorted
logreg.fit(X, y)
assorted_pred_class = logreg.predict(X)

# print the class predictions
assorted_pred_class

# plot the class predictions
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_class, color='red')

# What if we wanted the **predicted probabilities** instead of just the **class predictions**, to understand how confident we are in a given prediction?

# store the predicted probabilites of class 1
assorted_pred_prob = logreg.predict_proba(X)[:, 1]

# plot the predicted probabilities
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_prob, color='red')

# examine some example predictions
print logreg.predict_proba(1)
print logreg.predict_proba(2)
print logreg.predict_proba(3)

# What is this? The first column indicates the predicted probability of **class 0**, and the second column indicates the predicted probability of **class 1**.


# ## Part 4: Probability, odds, e, log, log-odds

# $$probability = \frac {one\ outcome} {all\ outcomes}$$
# $$odds = \frac {one\ outcome} {all\ other\ outcomes}$$

# Examples:
# - Dice roll of 1: probability = 1/6, odds = 1/5
# - Even dice roll: probability = 3/6, odds = 3/3 = 1
# - Dice roll less than 5: probability = 4/6, odds = 4/2 = 2

# $$odds = \frac {probability} {1 - probability}$$
# $$probability = \frac {odds} {1 + odds}$$

# create a table of probability versus odds
table = pd.DataFrame({'probability':[0.1, 0.2, 0.25, 0.5, 0.6, 0.8, 0.9]})
table['odds'] = table.probability/(1 - table.probability)
table

# What is **e**? It is the base rate of growth shared by all continually growing processes:

# exponential function: e^1
np.exp(1)

# What is a **(natural) log**? It gives you the time needed to reach a certain level of growth:

# time needed to grow 1 unit to 2.718 units
np.log(2.718)

# It is also the **inverse** of the exponential function:

np.log(np.exp(5))

# add log-odds to the table
table['logodds'] = np.log(table.odds)
table


# ## Part 5: What is Logistic Regression?

# **Linear regression:** continuous response is modeled as a linear combination of the features:
# $$y = \beta_0 + \beta_1x$$

# **Logistic regression:** log-odds of a categorical response being "true" (1) is modeled as a linear combination of the features:
# $$\log \left({p\over 1-p}\right) = \beta_0 + \beta_1x$$

# This is called the **logit function**.

# Probability is sometimes written as pi:
# $$\log \left({\pi\over 1-\pi}\right) = \beta_0 + \beta_1x$$

# The equation can be rearranged into the **logistic function**:
# $$\pi = \frac{e^{\beta_0 + \beta_1x}} {1 + e^{\beta_0 + \beta_1x}}$$

# In other words:
# - Logistic regression outputs the **probabilities of a specific class**
# - Those probabilities can be converted into **class predictions**

# The **logistic function** has some nice properties:
# - Takes on an "s" shape
# - Output is bounded by 0 and 1

# Notes:
# - **Multinomial logistic regression** is used when there are more than 2 classes.
# - Coefficients are estimated using **maximum likelihood estimation**, meaning that we choose parameters that maximize the likelihood of the observed data.

# ## Part 6: Interpreting Logistic Regression Coefficients

# plot the predicted probabilities again
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_prob, color='red')

# compute predicted log-odds for al=2 using the equation
logodds = logreg.intercept_ + logreg.coef_[0] * 2
logodds

# convert log-odds to odds
odds = np.exp(logodds)
odds

# convert odds to probability
prob = odds/(1 + odds)
prob

# compute predicted probability for al=2 using the predict_proba method
logreg.predict_proba(2)[:, 1]

# examine the coefficient for al
zip(feature_cols, logreg.coef_[0])

# **Interpretation:** A 1 unit increase in 'al' is associated with a 4.18 unit increase in the log-odds of 'assorted'.

# increasing al by 1 (so that al=3) increases the log-odds by 4.18
logodds = 0.64722323 + 4.1804038614510901
odds = np.exp(logodds)
prob = odds/(1 + odds)
prob

# compute predicted probability for al=3 using the predict_proba method
logreg.predict_proba(3)[:, 1]

# **Bottom line:** Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# examine the intercept
logreg.intercept_

# **Interpretation:** For an 'al' value of 0, the log-odds of 'assorted' is -7.71.

# convert log-odds to probability
logodds = logreg.intercept_
odds = np.exp(logodds)
prob = odds/(1 + odds)
prob

# That makes sense from the plot above, because the probability of assorted=1 should be very low for such a low 'al' value.

# Changing the $\beta_0$ value shifts the curve **horizontally**, whereas changing the $\beta_1$ value changes the **slope** of the curve.


# ## Part 7: Comparing Logistic Regression with Other Models

# Advantages of logistic regression:
# - Highly interpretable (if you remember how)
# - Model training and prediction are fast
# - No tuning is required (excluding regularization)
# - Features don't need scaling
# - Can perform well with a small number of observations
# - Outputs well-calibrated predicted probabilities

# Disadvantages of logistic regression:
# - Presumes a linear relationship between the features and the log-odds of the response
# - Performance is (generally) not competitive with the best supervised learning methods
# - Sensitive to irrelevant features
# - Can't automatically learn feature interactions
