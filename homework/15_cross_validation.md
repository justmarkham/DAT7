## Class 12 Pre-work: Cross-validation

Watch my video on [cross-validation](https://www.youtube.com/watch?v=6dbrR-WymjI) (36 minutes), and be prepared to **discuss it in class** on Wednesday. The [notebook](http://nbviewer.ipython.org/github/justmarkham/DAT7/blob/master/notebooks/12_cross_validation.ipynb) shown in the video is also in this repository.

Here are some questions to think about:

- What is the purpose of model evaluation?
- What do the terms training accuracy and testing accuracy mean?
- What is the drawback of training and testing on the same data?
- What is the drawback of train/test split?
- What is the role of "K" in K-fold cross-validation?
- When should you use train/test split, and when should you use cross-validation?
- Why do we pass X and y, not X_train and y_train, to the cross_val_score function?
- What is the point of the cross_val_score function's "scoring" parameter?
- What does cross_val_score do, in detail? What does it return?
- Under what circumstances does cross_val_score return negative scores?
