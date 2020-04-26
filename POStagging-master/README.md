POS tagger by Team GAP!

Team Members: Pranav Krishna SJ,Alagappan A, Ganesh Nalluru

This program learns to tag words based on the given train set which consists of tagged words and the
learning is tested on a test set without tags, finally generating a text file "pos-test-with-tags.txt"
consisting of tagged test words.

Accuracy:
    Baseline:
        If we consider baseline as assigning each word to the most frequent POS tag
        for the word, then the accuracy is around 82.3%.
    Bigram:
        The HMM POS tagger using a bigram model improved the accuracy from 82.3%
        to 83.1%
    Rules:
        Adding rules to the bigram HMM tagger improves the accuracy by 7%. The
        accuracy after creating rules by analyzing confusion matrix is 90.1%.
Rules:
    The set of rules used in this tagger are as follows,
    
    If the word is "a" and if the previous tags for the word does not denote
    end of sentences, then that word is a determiner.
    
    If the word is currently tagged as particle and if the word is not after different
    forms of verb then the word should be tagged as preposition or  subordinating conjunction.
    
    If the word is currently tagged as verb and if the previous tag is determiner,
    then the word should be tagged as noun.
               
    If the word is currently tagged as Wh-determiner and if the previous tag is not
    a noun then the word should be a preposition or  subordinating conjunction.
    
    If the word ends with "ing" then it is a gerund verb.
    
    If the first letter of the word is capitalized then it is a proper noun.
    
    If the word ends with 'able' then it is an adjective.
          

The algorithm for the tagger program is as follows.
Algorithm:
    Requests train and test files from the user.
    Extracts tags and count of each tag in the training file.
    Extracts words and list of tags for the word in the training set.
    Extracts bigrams of tags and count of the bigram in the training set.
    For each word in the test file:
        For each tag of the word in training file:
            Computes the probability of word given the tag. i.e, p(w|t)
            Computes the probability of tag given the previous tag.i.e, p(t|t-1)
            Multiples both these tags.
        Finds the tag with maximum probability and assigns it to to the word.
    A set of rules written after the analyzing the confusion matrix transforms the tags assigned.
'''
