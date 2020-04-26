
Sentiment classification using naive byes and logistic regression.


Command to run the file: python nb.py
The program should be inside the data directory Sentiment classification folder which has train and test files.

Different types of methods followed are as follows,
    Naive Bayes with stemming and word count
    Naive Bayes without stemming and word count
    Naive Bayes with stemming and binary count
    Naive Bayes without stemming and binary count
    Naive Bayes with stemming and TF-IDF 
    Naive Bayes without stemming and TF-IDF
    
Program flow:

      1. Main()
          The main function is used for contols. It runs 4 functions as mentioned below.

          1.1.nb()
          The funtion collects the count of non stemmed and stemmed vocabulary and assigns it to global variables.

              1.1.1.count_words()
              The funtion collects the training files. Tokenizes text into words. Creates stemmed vocabulary and 
              Counts the the occurance of each word in each class(positve and negative).

                  1.1.1.1 normalize_case()
                  Converts words with capitalized first letters in to lower case.

                  1.1.1.2 remove_tags()
                  Removes HTML tags

        1.2 get_test()
            The funtion collects the test data. Creates Bag of words and
            stemmed vocabulary.

        1.3 classify()
            The function uses the counts computed from previous step to calculate likelyhood and prior.
            Then the test document is classified based the result from above step.

        1.4 metrics()
            Prints the accuracy and confusion matrix

