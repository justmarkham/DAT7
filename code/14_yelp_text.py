'''
HOMEWORK: Yelp Review Text
'''

# TASK 1: read yelp.csv into a DataFrame
import pandas as pd
yelp = pd.read_csv('yelp.csv')

# TASK 2: create a new DataFrame that only contains the 5-star and 1-star reviews
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# TASK 3: split the new DataFrame into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(yelp_best_worst.text, yelp_best_worst.stars, random_state=1)

# TASK 4: use CountVectorizer to create document-term matrices from X_train and X_test
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
train_dtm = vect.fit_transform(X_train)
test_dtm = vect.transform(X_test)

# TASK 5: use Naive Bayes to predict the star rating for the testing set, and calculate accuracy
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, y_train)
y_pred_class = nb.predict(test_dtm)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

# TASK 6: calculate the AUC
y_pred_prob = nb.predict_proba(test_dtm)[:, 1]
import numpy as np
y_test_binary = np.where(y_test==5, 1, 0)
print metrics.roc_auc_score(y_test_binary, y_pred_prob)

# TASK 7: plot the ROC curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test_binary, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# TASK 8: print the confusion matrix, and calculate sensitivity and specificity
print metrics.confusion_matrix(y_test, y_pred_class)
813 / float(25 + 813)   # sensitivity
126 / float(126 + 58)   # specificity

# TASK 9: browse the review text for the false positive and false negatives
X_test[y_test < y_pred_class]   # false positives
X_test[y_test > y_pred_class]   # false negatives

# TASK 10: change the threshold to balance sensitivity and specificity
y_pred_class = np.where(y_pred_prob > 0.999, 5, 1)
print metrics.confusion_matrix(y_test, y_pred_class)
723 / float(115 + 723)  # sensitivity
162 / float(162 + 22)   # specificity

# TASK 11 (BONUS): 5-class classification on the original DataFrame
X_train, X_test, y_train, y_test = train_test_split(yelp.text, yelp.stars, random_state=1)
train_dtm = vect.fit_transform(X_train)
test_dtm = vect.transform(X_test)
nb.fit(train_dtm, y_train)
y_pred_class = nb.predict(test_dtm)
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.confusion_matrix(y_test, y_pred_class)
