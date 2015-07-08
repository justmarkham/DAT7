## Class 12 Pre-work: Cross-validation

Watch my video on [cross-validation](https://www.youtube.com/watch?v=6dbrR-WymjI) (36 minutes), and be prepared to **discuss it in class** on Wednesday. The [notebook](http://nbviewer.ipython.org/github/justmarkham/DAT7/blob/master/notebooks/12_cross_validation.ipynb) shown in the video is also in this repository.

Here are some questions to think about:

- What is the purpose of model evaluation?
    - The purpose is to estimate the likely performance of a model on out-of-sample data, so that we can choose the model that is most likely to generalize, and so that we can have an idea of how well that model will actually perform.
- What do the terms training accuracy and testing accuracy mean?
    - Training accuracy is the accuracy achieved when training and testing on the same data. Testing accuracy is the accuracy achieved when splitting the data with train/test split, training the model on the training set, and testing the model on the testing set.
- What is the drawback of training and testing on the same data?
    - Training accuracy is maximized for overly complex models which overfit the training data, and thus it's not a good measure of how well a model will generalize.
- What is the drawback of train/test split?
    - Testing accuracy can change a lot depending upon which observations happen to be in the training and testing sets.
- What is the role of "K" in K-fold cross-validation?
    - K is the number of partitions that the dataset is split into, and thus is the number of iterations that cross-validation will run for.
- When should you use train/test split, and when should you use cross-validation?
    - Train/test split is useful when you want to inspect your testing results (via confusion matrix or ROC curve) and when evaluation speed is a concern. Cross-validation is useful when you are most concerned with the accuracy of your estimation.
- Why do we pass X and y, not X_train and y_train, to the cross_val_score function?
    - cross_val_score will take care of splitting the data into the K folds, so we don't need to split it ourselves.
- What is the point of the cross_val_score function's "scoring" parameter?
    - cross_val_score needs to know what evaluation metric to calculate, since many different metrics are available.
- What does cross_val_score do, in detail? What does it return?
    - First, it splits the data into K equal folds. Then, it trains the model on folds 2 through K, tests the model on fold 1, and calculates the requested evaluation metric. Then, it repeats that process K-1 more times, until every fold has been the testing set exactly once. It returns a NumPy array containing the K scores.
- Under what circumstances does cross_val_score return negative scores?
    - The scores will be negative if the evaluation metric is a loss function (something you want to minimize) rather than a reward function (something you want to maximize).
