# # Linear Regression
# *Adapted from Chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)*

# # Part 1: Introduction
# - **Classification problem:** supervised learning problem with a categorical response
# - **Regression problem**: supervised learning problem with a continuous response
# - **Linear regression:** machine learning model that can be used for regression problems

# Why are we learning linear regression?
# - widely used
# - runs fast
# - easy to use (no tuning is required)
# - highly interpretable
# - basis for many other methods

# Lesson goals:
# - Conceptual understanding of linear regression and how it "works"
# - Familiarity with key terminology
# - Ability to apply linear regression to a machine learning problem using scikit-learn
# - Ability to interpret model coefficients
# - Familiarity with different approaches for feature selection
# - Understanding of three different evaluation metrics for regression
# - Understanding of linear regression's strengths and weaknesses

# ## Libraries
# - [Statsmodels](http://statsmodels.sourceforge.net/): "statistics in Python"
#     - robust functionality for linear modeling
#     - useful for teaching purposes
#     - will not be used in the course outside of this lesson
# - [scikit-learn](http://scikit-learn.org/stable/): "machine learning in Python"
#     - significantly more functionality for general purpose machine learning

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# ## Reading the advertising data

# read data into a DataFrame
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()

# What are the observations?
# - Each observation represents **one market** (200 markets in the dataset)

# What are the features?
# - **TV:** advertising dollars spent on TV for a single product (in thousands of dollars)
# - **Radio:** advertising dollars spent on Radio
# - **Newspaper:** advertising dollars spent on Newspaper

# What is the response?
# - **Sales:** sales of a single product in a given market (in thousands of widgets)

# ## Questions about the data
# You are asked by the company: On the basis of this data, how should we spend our advertising money in the future?
# You come up with more specific questions:
# 1. Is there a relationship between ads and sales?
# 2. How strong is that relationship?
# 3. Which ad types contribute to sales?
# 4. What is the effect of each ad type of sales?
# 5. Given ad spending in a particular market, can sales be predicted?

# ## Visualizing the data

# Use a **scatter plot** to visualize the relationship between the features and the response.

# scatter plot in Seaborn
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=6, aspect=0.7)

# include a "regression line"
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=6, aspect=0.7, kind='reg')

# scatter plot in Pandas
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 6))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])

# Use a **scatter matrix** to visualize the relationship between all numerical variables.

# scatter matrix in Seaborn
sns.pairplot(data)

# scatter matrix in Pandas
pd.scatter_matrix(data, figsize=(12, 10))

# Use a **correlation matrix** to visualize the correlation between all numerical variables.

# compute correlation matrix
data.corr()

# display correlation matrix in Seaborn using a heatmap
sns.heatmap(data.corr())

# # Part 2: Simple linear regression

# Simple linear regression is an approach for predicting a **continuous response** using a **single feature**. It takes the following form:
# $y = \beta_0 + \beta_1x$
# - $y$ is the response
# - $x$ is the feature
# - $\beta_0$ is the intercept
# - $\beta_1$ is the coefficient for x

# $\beta_0$ and $\beta_1$ are called the **model coefficients**:

# - We must "learn" the values of these coefficients to create our model.
# - And once we've learned these coefficients, we can use the model to predict Sales.

# ## Estimating ("learning") model coefficients
# - Coefficients are estimated during the model fitting process using the **least squares criterion**.
# - We are find the line (mathematically) which minimizes the **sum of squared residuals** (or "sum of squared errors").

# In this diagram:
# - The black dots are the **observed values** of x and y.
# - The blue line is our **least squares line**.
# - The red lines are the **residuals**, which are the distances between the observed values and the least squares line.

# How do the model coefficients relate to the least squares line?
# - $\beta_0$ is the **intercept** (the value of $y$ when $x$=0)
# - $\beta_1$ is the **slope** (the change in $y$ divided by change in $x$)

# Let's estimate the model coefficients for the advertising data:

### STATSMODELS ###

# create a fitted model
lm = smf.ols(formula='Sales ~ TV', data=data).fit()

# print the coefficients
lm.params

### SCIKIT-LEARN ###

# create X and y
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_

# ## Interpreting model coefficients

# How do we interpret the TV coefficient ($\beta_1$)?
# - A "unit" increase in TV ad spending is **associated with** a 0.0475 "unit" increase in Sales.
# - Meaning: An additional $1,000 spent on TV ads is **associated with** an increase in sales of 47.5 widgets.
# - This is not a statement of **causation**.

# If an increase in TV ad spending was associated with a **decrease** in sales, $\beta_1$ would be **negative**.

# ## Using the model for prediction

# Let's say that there was a new market where the TV advertising spend was **$50,000**. What would we predict for the Sales in that market?
# $$y = \beta_0 + \beta_1x$$
# $$y = 7.0326 + 0.0475 \times 50$$

# manually calculate the prediction
7.0326 + 0.0475*50

### STATSMODELS ###

# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'TV': [50]})

# predict for a new observation
lm.predict(X_new)

### SCIKIT-LEARN ###

# predict for a new observation
linreg.predict(50)

# Thus, we would predict Sales of **9,409 widgets** in that market.

# ## Does the scale of the features matter?

# Let's say that TV was measured in dollars, rather than thousands of dollars. How would that affect the model?

data['TV_dollars'] = data.TV * 1000
data.head()

### SCIKIT-LEARN ###

# create X and y
feature_cols = ['TV_dollars']
X = data[feature_cols]
y = data.Sales

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_

# How do we interpret the TV_dollars coefficient ($\beta_1$)?
# - A "unit" increase in TV ad spending is **associated with** a 0.0000475 "unit" increase in Sales.
# - Meaning: An additional dollar spent on TV ads is **associated with** an increase in sales of 0.0475 widgets.
# - Meaning: An additional $1,000 spent on TV ads is **associated with** an increase in sales of 47.5 widgets.

# predict for a new observation
linreg.predict(50000)

# The scale of the features is **irrelevant** for linear regression models, since it will only affect the scale of the coefficients, and we simply change our interpretation of the coefficients.

# # Part 3: A deeper understanding

# ## Bias and variance

# Linear regression is a low variance/high bias model:
# - **Low variance:** Under repeated sampling from the underlying population, the line will stay roughly in the same place
# - **High bias:** The line will rarely fit the data well

# A closely related concept is **confidence intervals**.

# ## Confidence intervals

# Statsmodels calculates 95% confidence intervals for our model coefficients, which are interpreted as follows: If the population from which this sample was drawn was **sampled 100 times**, approximately **95 of those confidence intervals** would contain the "true" coefficient.

### STATSMODELS ###

# print the confidence intervals for the model coefficients
lm.conf_int()

# - We only have a **single sample of data**, and not the **entire population of data**.
# - The "true" coefficient is either within this interval or it isn't, but there's no way to actually know.
# - We estimate the coefficient with the data we do have, and we show uncertainty about that estimate by giving a range that the coefficient is **probably** within.
# - From Quora: [What is a confidence interval in layman's terms?](http://www.quora.com/What-is-a-confidence-interval-in-laymans-terms/answer/Michael-Hochster)

# Note: 95% confidence intervals are just a convention. You can create 90% confidence intervals (which will be more narrow), 99% confidence intervals (which will be wider), or whatever intervals you like.

# A closely related concept is **hypothesis testing**.

# ## Hypothesis testing and p-values

# General process for hypothesis testing:
# - You start with a **null hypothesis** and an **alternative hypothesis** (that is opposite the null).
# - You check whether the data supports **rejecting the null hypothesis** or **failing to reject the null hypothesis**.

# For model coefficients, here is the conventional hypothesis test:
# - **null hypothesis:** There is no relationship between TV ads and Sales (and thus $\beta_1$ equals zero)
# - **alternative hypothesis:** There is a relationship between TV ads and Sales (and thus $\beta_1$ is not equal to zero)

# How do we test this hypothesis?
# - The **p-value** is the probability that the relationship we are observing is occurring purely by chance.
# - If the 95% confidence interval for a coefficient **does not include zero**, the p-value will be **less than 0.05**, and we will reject the null (and thus believe the alternative).
# - If the 95% confidence interval **includes zero**, the p-value will be **greater than 0.05**, and we will fail to reject the null.

### STATSMODELS ###

# print the p-values for the model coefficients
lm.pvalues

# Thus, a p-value less than 0.05 is one way to decide whether there is **likely** a relationship between the feature and the response. In this case, the p-value for TV is far less than 0.05, and so we **believe** that there is a relationship between TV ads and Sales.

# Note that we generally ignore the p-value for the intercept.

# ## How well does the model fit the data?

# R-squared:
# - A common way to evaluate the overall fit of a linear model
# - Defined as the **proportion of variance explained**, meaning the proportion of variance in the observed data that is explained by the model
# - Also defined as the reduction in error over the **null model**, which is the model that simply predicts the mean of the observed response
# - Between 0 and 1, and higher is better

# Here's an example of what R-squared "looks like":

# Let's calculate the R-squared value for our simple linear model:

### STATSMODELS ###

# print the R-squared value for the model
lm.rsquared

### SCIKIT-LEARN ###

# calculate the R-squared value for the model
y_pred = linreg.predict(X)
metrics.r2_score(y, y_pred)

# - The threshold for a **"good" R-squared value** is highly dependent on the particular domain.
# - R-squared is more useful as a tool for **comparing models**.

# # Part 4: Multiple Linear Regression

# Simple linear regression can easily be extended to include multiple features, which is called **multiple linear regression**:
# $y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$

# Each $x$ represents a different feature, and each feature has its own coefficient:
# $y = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio + \beta_3 \times Newspaper$

### SCIKIT-LEARN ###

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_

# pair the feature names with the coefficients
zip(feature_cols, linreg.coef_)

# For a given amount of Radio and Newspaper spending, an increase of $1000 in **TV** spending is associated with an **increase in Sales of 45.8 widgets**.
# For a given amount of TV and Newspaper spending, an increase of $1000 in **Radio** spending is associated with an **increase in Sales of 188.5 widgets**.
# For a given amount of TV and Radio spending, an increase of $1000 in **Newspaper** spending is associated with an **decrease in Sales of 1.0 widgets**. How could that be?

# ## Feature selection

# How do I decide **which features to include** in a linear model?

# ### Using p-values

# We could try a model with all features, and only keep features in the model if they have **small p-values**:

### STATSMODELS ###

# create a fitted model with all three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

# print the p-values for the model coefficients
print lm.pvalues

# This indicates we would reject the null hypothesis for **TV and Radio** (that there is no association between those features and Sales), and fail to reject the null hypothesis for **Newspaper**. Thus, we would keep TV and Radio in the model.

# However, this approach has **drawbacks**:
# - Linear models rely upon a lot of **assumptions** (such as the features being independent), and if those assumptions are violated (which they usually are), p-values are less reliable.
# - Using a p-value cutoff of 0.05 means that if you add 100 features to a model that are **pure noise**, 5 of them (on average) will still be counted as significant.

# ### Using R-squared
# We could try models with different sets of features, and **compare their R-squared values**:

# R-squared value for the model with two features
lm = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
lm.rsquared

# R-squared value for the model with three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lm.rsquared

# This would seem to indicate that the best model includes **all three features**. Is that right?
# - R-squared will always increase as you add more features to the model, even if they are **unrelated** to the response.
# - As such, using R-squared as a model evaluation metric can lead to **overfitting**.
# - **Adjusted R-squared** is an alternative that penalizes model complexity (to control for overfitting), but it generally [under-penalizes complexity](http://scott.fortmann-roe.com/docs/MeasuringError.html).

# As well, R-squared depends on the same assumptions as p-values, and it's less reliable if those assumptions are violated.

# ### Using train/test split (or cross-validation)

# A better approach to feature selection!
# - They attempt to directly estimate how well your model will **generalize** to out-of-sample data.
# - They rely on **fewer assumptions** that linear regression.
# - They can easily be applied to **any model**, not just linear models.

# ## Evaluation metrics for regression problems

# Evaluation metrics for classification problems, such as **accuracy**, are not useful for regression problems. We need evaluation metrics designed for comparing **continuous values**.
# Let's create some example numeric predictions, and calculate three common evaluation metrics for regression problems:

# define true and predicted response values
y_true = [100, 50, 30, 20]
y_pred = [90, 50, 50, 30]

# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
print metrics.mean_absolute_error(y_true, y_pred)

# **Mean Squared Error** (MSE) is the mean of the squared errors:
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
print metrics.mean_squared_error(y_true, y_pred)

# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
print np.sqrt(metrics.mean_squared_error(y_true, y_pred))

# Comparing these metrics:
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.

# All of these are **loss functions**, because we want to minimize them.

# Here's an additional example, to demonstrate how MSE/RMSE punish larger errors:

# same true values as above
y_true = [100, 50, 30, 20]

# new set of predicted values
y_pred = [60, 50, 30, 20]

# MAE is the same as before
print metrics.mean_absolute_error(y_true, y_pred)

# RMSE is larger than before
print np.sqrt(metrics.mean_squared_error(y_true, y_pred))

# ## Using train/test split for feature selection
# Let's use train/test split with RMSE to decide whether Newspaper should be kept in the model:

# define a function that accepts X and y and computes testing RMSE
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# include Newspaper
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
train_test_rmse(X, y)

# exclude Newspaper
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
train_test_rmse(X, y)

# ## Comparing linear regression with other models

# Advantages of linear regression:
# - Simple to explain
# - Highly interpretable
# - Model training and prediction are fast
# - No tuning is required (excluding regularization)
# - Features don't need scaling
# - Can perform well with a small number of observations

# Disadvantages of linear regression:
# - Presumes a linear relationship between the features and the response
# - Performance is (generally) not competitive with the best supervised learning methods due to high bias
# - Sensitive to irrelevant features
# - Can't automatically learn feature interactions
