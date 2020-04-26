'''
Author: Alagappan, Ganesh, Pranav, Tejasvi
Description: AIT 726 Homework 1
Date: 09/29/2019

Command to run the file: python LR.py

The program should be inside the data directory Sentiment Classification Data->test, Sentiment Classification Data->Train
main()
    The main function has 3 parameters and is run inside a loop, the parameters are explained inside the main function.
    The data is read using the create_list() function which returns the positive and negative list of texts.
    The list is sent inside the CreateDataFrame() function where it shuffles data and returns the text and target variable
    Based on the input parameter the text is stemmed or not stemmed followed by creating a vocabulary list
    An empty trainVector is created based on nos_of_documents * len(vocabulary_list) and is vectorised based on the input parameters

    Logistic Regression model
        Train Set
            Concatenating 1's to the trainVector for the bias variable
            Initialise the theta to zero
            Initialise the iterations to 1000
            Initialise the learning rate to 0.01

        Based on the user input use the regularised cost function or the normal function

        Test set
            Vectorize the test set based on the vocabulary created using the train set
            Concatenating 1's to the testVector for the bias variable
            Predicting the testVector based on the optimised theta values
            The values is set as 1 if the predicted value is >0.5 and 0 if <=0.5
            Printing the final Output

Function flow
    Train Set
    ---------
    Train Set positve
        main()-> create_list()
    Train Set Negative
        main()-> create_list()

        main()-> createDataFrame()


    if parameter1=1
        main()-> cleanStemmed()
    else
        main()-> clean()


    if parameter2=1
        main()->vectorizer(TFIDF)

    else if parameter2=2
        main()->vectorizer(Count Vectorizer)
    else()
        main()->vectorizer(Binary Vectorizer)


    if parameter3=1
        main()-> gradient_descent_reg()->sigmoid()->compute_cost_reg()
    else
        main()-> gradient_descent()->sigmoid()->compute_cost()


    Test Set
    -------

    Test Set positve
        main()-> create_list()
    Test Set Neagtive
        main()-> create_list()

        main()-> createDataFrame()


    if parameter1=1
        main()-> cleanStemmed()
    else
        main()-> clean()


    if parameter2=1
        main()->vectorizer(TFIDF)
    else if parameter2=2
        main()->vectorizer(Count Vectorizer)
    else()
        main()->vectorizer(Binary Vectorizer)


    main()->predict()

Result
Data Clean:  Stemmed
Vectorization:   TF-IDF vectorizer
LR cost function:  Regularized
F1 SCore:  0.7935950413223141
Accuracy:  0.7937937937937938
Confusion Matrix:
[[381 119]
 [ 87 412]]

Data Clean:  Stemmed
Vectorization:   TF-IDF vectorizer
LR cost function:  Not regularized
F1 SCore:  0.7935950413223141
Accuracy:  0.7937937937937938
Confusion Matrix:
[[381 119]
 [ 87 412]]

Data Clean:  Stemmed
Vectorization:   Count vectorizer
LR cost function:  Regularized
F1 SCore:  0.7377311678837968
Accuracy:  0.7377377377377378
Confusion Matrix:
[[371 129]
 [133 366]]

Data Clean:  Stemmed
Vectorization:   Count vectorizer
LR cost function:  Not regularized
F1 SCore:  0.7377311678837968
Accuracy:  0.7377377377377378
Confusion Matrix:
[[371 129]
 [133 366]]

Data Clean:  Stemmed
Vectorization:  Binary vectorizer
LR cost function:  Regularized
F1 SCore:  0.808636236449467
Accuracy:  0.8088088088088088
Confusion Matrix:
[[389 111]
 [ 80 419]]

Data Clean:  Stemmed
Vectorization:  Binary vectorizer
LR cost function:  Not regularized
F1 SCore:  0.808636236449467
Accuracy:  0.8088088088088088
Confusion Matrix:
[[389 111]
 [ 80 419]]

Data Clean:  Not Stemmed
Vectorization:   TF-IDF vectorizer
LR cost function:  Regularized
F1 SCore:  0.7996936254852267
Accuracy:  0.7997997997997998
Confusion Matrix:
[[388 112]
 [ 88 411]]

Data Clean:  Not Stemmed
Vectorization:   TF-IDF vectorizer
LR cost function:  Not regularized
F1 SCore:  0.7996936254852267
Accuracy:  0.7997997997997998
Confusion Matrix:
[[388 112]
 [ 88 411]]

Data Clean:  Not Stemmed
Vectorization:   Count vectorizer
LR cost function:  Regularized
F1 SCore:  0.7557555110220441
Accuracy:  0.7557557557557557
Confusion Matrix:
[[377 123]
 [121 378]]

Data Clean:  Not Stemmed
Vectorization:   Count vectorizer
LR cost function:  Not regularized
F1 SCore:  0.7557555110220441
Accuracy:  0.7557557557557557
Confusion Matrix:
[[377 123]
 [121 378]]

Data Clean:  Not Stemmed
Vectorization:  Binary vectorizer
LR cost function:  Regularized
F1 SCore:  0.8106164936603351
Accuracy:  0.8108108108108109
Confusion Matrix:
[[389 111]
 [ 78 421]]

Data Clean:  Not Stemmed
Vectorization:  Binary vectorizer
LR cost function:  Not regularized
F1 SCore:  0.8106164936603351
Accuracy:  0.8108108108108109
Confusion Matrix:
[[389 111]
 [ 78 421]]

Conclusion
    Not Stemmed Binary representation gives the highest accuracy 0.8106164936603351
    Count Vectorizer lowers the accuracy to 0.7557555110220441
    While TFIDF performs better than Count Vectorizer but lesser than Binary representaion 0.7996936254852267
    Regularised logistic regression doesn't affect the result as the learning rate is fixed

'''




import time
start_time = time.time()
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import re
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import warnings
from sklearn import preprocessing
warnings.filterwarnings("ignore")
#import logging
#import logging.config

#reading the data
pos_train = os.listdir("train/pos/")
neg_train = os.listdir("train/neg/")
pos_test = os.listdir("test/pos/")
neg_test = os.listdir("test/neg/")

#Removing HTML tags
def remove_tags(text):
    s = re.sub(r'<[^>]+>', '', text)
    return s

#https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
# Removing Emoticons
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Parsing and cleaning the data
def create_list(dir,type,type1):
    return_list=[]
    # reduced data size-running out of memory
    for i in range(0, 500):
        file1 = open(type+"/"+type1+"/" + dir[i])
        try:
            text = file1.read()
            text = remove_tags(text)
            text = remove_emoji(text)
            return_list.append(text)
        except UnicodeDecodeError:
            k=0
        file1.close()

    return return_list

# Data Frame is generated with the passed pos and neg text
# Pos text are coded as 1 and neg text are coded as 0
# Finally the data is shuffled
def createDataFrame(pos_train_list,neg_train_list):
    df = pd.DataFrame(pos_train_list)
    target1 = [1] * len(pos_train_list)
    df["target"] = target1
    df = df.rename(columns={0: "text"})

    df1 = pd.DataFrame(neg_train_list)
    target2 = [0] * len(neg_train_list)
    df1["target"] = target2
    df1 = df1.rename(columns={0: "text"})

    data = pd.concat([df, df1])
    data = shuffle(data,random_state=9)
    x = list(data["text"])
    y=np.array(data["target"])

    return x,y;

# Tokenizing the text
def clean(text):

    vocab=[]
    for j in word_tokenize(text):
        if (j != ''):
            if not j.islower() and not j.isupper():
                j = j.lower()
            vocab.append(j)

    return vocab

# Tokenizing the text and stemming it
def cleanStemmed(text):

    ps = PorterStemmer()
    vocab_stemmed=[]
    for j in word_tokenize(text):
        if (j != ''):
            if not j.islower() and not j.isupper():
                j = j.lower()
            vocab_stemmed.append(ps.stem(j))

    return vocab_stemmed

# sigmoind function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Running the regularised gradient function in a loop till the max iterations is reached
def gradient_descent_reg(X, y, params, learning_rate, iterations, lmbda):

    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
        cost_history[i] = compute_cost_reg(X, y, params, lmbda)

    return (cost_history, params)

# The objective regularised cost function
def compute_cost_reg(X, y, theta, lmbda):

    m = len(y)
    h = sigmoid(X @ theta)
    temp = theta
    cost = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) + (lmbda / (2 * m)) * np.sum(np.square(temp))

    return cost

# Running the gradient function in a loop till the max iterations is reached
def gradient_descent(X, y, params, learning_rate, iterations):

    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
        cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)


# The objective cost function
def compute_cost(X, y, theta):

    m = len(y)
    h = sigmoid(X @ theta)
    cost = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))

    return cost

# Final prediction function
def predict(X, params):
    return np.round(sigmoid(X @ params))

# Vectorizing based user input
def vectorizer(X,vectorArr,dict_vocab,vectorType,row,col):

    if (vectorType==1):
        for i in range(0, len(X)):
            for j in X[i]:
                if j in dict_vocab:
                    vectorArr[i, dict_vocab[j]] += 1


        idf= np.zeros((row, col), dtype=np.int64)
        for i in range(0,len(vectorArr)):
            for j in range(0,col):
                if vectorArr[i][j] > 0:
                    idf[i][j]= math.log10(row / float(vectorArr[i][j]))

                else:
                    idf[i][j]=0
        vectorArr=np.multiply(vectorArr, idf)



    elif (vectorType==2):
        for i in range(0, len(X)):
            for j in X[i]:
                if j in dict_vocab:
                    vectorArr[i, dict_vocab[j]] += 1


    else:
        for i in range(0, len(X)):
            for j in X[i]:
                if j in dict_vocab:
                    vectorArr[i,dict_vocab[j]]=1

    return vectorArr


def main(stemmed,vectorType,regularized):
    
#    logging.basicConfig(filename='logl.log',filemode='w',level=logging.INFO,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    
    # loading the train set
    pos_train_list = create_list(pos_train,"train", "pos")
    neg_train_list = create_list(neg_train,"train", "neg")
    X,y=createDataFrame(pos_train_list,neg_train_list)

    # Stemming/ Not Stemming data
    if stemmed == 1:
        for i in range(0, len(X)):
            X[i] = cleanStemmed(X[i])

    else:
        for i in range(0, len(X)):
            X[i] = clean(X[i])

    # Creating vocabulary list
    vocab = X[0]
    for i in range(1, len(X)):
        vocab.extend(X[i])
    vocab = sorted(set(vocab))

    row = len(X)
    col = len(vocab)

    dict_vocab = {}
    for i, j in enumerate(vocab):
        dict_vocab[j] = i
    trainVector = np.zeros((row, col), dtype=np.int64)

    # Vectorizing the trainset
    trainVector=vectorizer(X,trainVector,dict_vocab,vectorType,row,col)
    m, n = trainVector.shape
    trainVector = np.concatenate([np.ones((m, 1)), trainVector], axis=1)
    trainVector = preprocessing.scale(trainVector)

    initial_theta = np.zeros(n + 1)
    iterations = 1000
    learning_rate = 0.01

    # Logistic function
    if regularized==1:
        lmbda = 0.1
        (cost_history, params_optimal) = gradient_descent_reg(trainVector, y, initial_theta, learning_rate, iterations,lmbda)


    else :
        (cost_history, params_optimal) = gradient_descent(trainVector, y, initial_theta, learning_rate, iterations)

    # Loading the test set
    pos_test_list = create_list(pos_test, "test", "pos")
    neg_test_list = create_list(neg_test, "test", "neg")
    X_test, y_test = createDataFrame(pos_test_list, neg_test_list)

    # Stemming data
    if (stemmed == 1):
        for i in range(0, len(X_test)):
            X_test[i] = cleanStemmed(X_test[i])

    else:
        for i in range(0, len(X_test)):
            X_test[i] = clean(X_test[i])

    row = len(X_test)
    col = len(vocab)

    testVector = np.zeros((row, col), dtype=np.int64)
    # Vectorizing the test data
    testVector = vectorizer(X_test,testVector,dict_vocab,vectorType,row,col)
    m, n = testVector.shape

    testVector=np.concatenate([np.ones((m, 1)), testVector], axis=1)
    testVector = preprocessing.scale(testVector)

    # Final Prediction
    preds = predict(testVector , params_optimal)

    # Final values based on threshold value 0.5
    for i in range(0, len(preds)):
        if (preds[i] <= 0.5):
            preds[i] = 0
        else:
            preds[i] = 1

    if(stemmed==1):
        dataClean="Stemmed"
    else:
        dataClean = "Not Stemmed"

    if (vectorType == 1):
        type = " TF-IDF vectorizer"
    elif (vectorType==2):
        type = " Count vectorizer"
    else:
        type="Binary vectorizer"

    if (regularized==1):
        reg="Regularized"
    else:
        reg="Not regularized"

    # Output
    print("Data Clean: ",dataClean)
#    logging.info("Data Clean: ",dataClean)
    
    print("Vectorization: ",type)
#    logging.info("Vectorization: ",type)
    
    print("LR cost function: ",reg)
#    logging.info("LR cost function: ",reg)

    print("F1 SCore: ",f1_score(y_test,preds, average='macro'))
#    logging.info("F1 SCore: ",f1_score(y_test,preds, average='macro'))
    
    print("Accuracy: ",accuracy_score(y_test,preds))
#    logging.info("Accuracy: ",accuracy_score(y_test,preds))
    
    print("Confusion Matrix:")
#    logging.info("Confusion Matrix:")
    
    print(confusion_matrix(y_test, preds))
#    logging.info(confusion_matrix(y_test, preds))
    
    print(" ")
#    logging.info(" ")

if __name__ == "__main__" :
    #Parameters
        #First Parameter
            #stemmed:1 
            # Not Stemmed: any other number
        #Second Parameter
            #TF-IDF:1
            #Count: 2
            #Binary : any other number
        #Third Parameter
            #regularized=1
            #not regularized=any other number
    #'''

    
    for i in range(1, 3):
        for j in range(1,4):
            for k in range(1,3):
                main(i,j,k)
    #'''
    #main(0,2,1)
    #main(0, 2, 0)

    
    print("--- %s seconds ---" % (time.time() - start_time))
#    logging.info("--- %s seconds ---" % (time.time() - start_time))
