## Class 14 Pre-work: Spam Filtering

Read Paul Graham's [A Plan for Spam](http://www.paulgraham.com/spam.html) and be prepared to **discuss it in class on Wednesday**.

Here are some questions to think about while you read:
* Should a spam filter optimize for sensitivity or specificity, in Paul's opinion?
    * specificity to minimize false positives
* Before he tried the "statistical approach" to spam filtering, what was his approach?
    * hand engineering features and computing a "score"
* What are the key components of his statistical filtering system? In other words, how does it work?
    * scan the entire text (including headers) and tokenize it
    * count number of occurrences of each token in ham corpus and spam corpus
    * assign each token a spam score based upon its relative frequency
    * for new mail, only take 15 most interesting tokens into account
* What did Paul say were some of the benefits of the statistical approach?
    * it works better (almost no false positives)
    * less work for him because it discovers features automatically
    * you know what the "score" means
    * can easily be tuned to the individual user
    * evolves with the spam
    * creates an implicit whitelist/blacklist of email addresses, server names, etc.
* How good was his prediction of the "spam of the future"?
    * great!
