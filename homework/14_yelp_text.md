## Class 14 Homework: Yelp Review Text

This assignment uses the same data as the [class 10 homework](10_yelp_reviews.md). This time, we will attempt to classify reviews as either 5-star or 1-star using only the review text!

After each task, I recommend that you check the **shape** and the **contents** of your objects, to confirm that they match your expectations.

**Tasks:**

1. Read `yelp.csv` into a DataFrame.
2. Create a new DataFrame that only contains the 5-star and 1-star reviews.
3. Split the new DataFrame into training and testing sets, using the review text as the feature and the star rating as the response.
4. Use CountVectorizer to create document-term matrices from X_train and X_test.
    * **Hint:** If you run into a decoding error, instantiate the vectorizer with the argument `decode_error='ignore'`.
5. Use Naive Bayes to predict the star rating for reviews in the testing set, and calculate the accuracy.
6. Calculate the AUC.
    * **Hint 1:** Make sure to pass the predicted probabilities to `roc_auc_score`, not the predicted classes.
    * **Hint 2:** `roc_auc_score` will get confused if y_test contains fives and ones, so you will need to create a new object that contains ones and zeros instead.
7. Plot the ROC curve.
8. Print the confusion matrix, and calculate the sensitivity and specificity.
9. Browse through the review text for some of the false positives and false negatives. Based on your knowledge of how Naive Bayes works, do you have any theories about why the model is incorrectly classifying these reviews?
10. Let's pretend that you want to balance sensitivity and specificity. You can achieve this by changing the threshold for predicting a 5-star review. What threshold approximately balances sensitivity and specificity?
11. **Bonus:** Turn this into a 5-class classification problem by predicting the star rating using the original DataFrame (from step 1). Calculate the accuracy and print the confusion matrix. Comment on the results.
