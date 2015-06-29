## Class 9 Exercise: Glass Identification

Let's practice what we've learned using the [Glass Identification dataset](http://archive.ics.uci.edu/ml/datasets/Glass+Identification).

1. Read the data into a DataFrame.
2. Briefly explore the data to make sure the DataFrame matches your expectations.
3. Let's convert this into a binary classification problem. Create a new DataFrame column called "binary":
    * If type of glass = 1/2/3/4, set binary = 0.
    * If type of glass = 5/6/7, set binary = 1.
4. Create a feature matrix "X" using all features. (Think carefully about which columns are actually features!)
5. Create a response vector "y" from the "binary" column.
6. Split X and y into training and testing sets.
7. Fit a KNN model on the training set using K=5.
8. Make predictions on the testing set and calculate testing accuracy.
9. Write a for loop that computes the testing accuracy for a range of K values.
10. Plot the K value versus testing accuracy to help you choose an optimal value for K.
11. Calculate the testing accuracy that could be achieved by always predicting the most frequent class in the testing set. (This is known as the "null accuracy".)
12. **Bonus:** Explore the data to determine which features look like good predictors, and then redo this exercise using only those features to see if you can achieve a higher testing accuracy!
