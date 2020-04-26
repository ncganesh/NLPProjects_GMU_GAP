# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:53:14 2019

@author: Pranav Krishna
"""

import os
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
import re
import numpy as np
import nltk
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def normalize_case(s):    
    '''
    Paramaeter: Word to be normalized
    Converts words with capitalized first letters in to lower case.
    '''
    if(not s.isupper()):
        return s.lower()
    else:
        return s
    
def remove_tags(s):
    '''
    Paramaeter: Word to be normalized
    Removes HTML tags
    '''
    s = re.sub(r'[^\w\s]', '', s)
    return s

def count_words(rootdir):
    '''
    Parameter: root directory
    
    The funtion collects the training files. Tokenizes text into words. Creates stemmed vocabulary and 
    Counts the the occurance of each word in each class(positve and negative).
    '''    
    #Port Stemmer for stemming vocabulary
    ps = PorterStemmer()
    vocab=[]
    total_words=0
    stemmed_vocab=[]
    prior=0
    binary_count={}
    stemmed_binary_count={}
    bigram=[]
    cleaned_document=[]
    #For each directory in the path
    z=0
    for subdir, dirs, files in os.walk(rootdir):
        #For each file in the directory
        
        for file in files:
            z=z+1
            if(z>1000):
                break
            cleaned_document=[]
            f=open(rootdir+file,'r',encoding='utf-8') #use the absolute URL of the file
            lines = f.readlines()
            #For each line in the file
            for line in lines:
                document=word_tokenize(line)
                for i in range(0,len(document)):
                    #Normalize case for the word
                    document[i]=normalize_case(document[i])
                    #Remove HTML tags
                    document[i]=remove_tags(document[i])
                    if(document[i]!=''):
                        cleaned_document.append(document[i])
                        vocab.append(document[i])
                #print(line)
                
            bigram.extend(list(nltk.bigrams(cleaned_document)))
            
    return bigram,vocab
            
            
            
rootdir= r'train\pos\\' 
pos_bigram,vocab=count_words(rootdir)

neg_bigram=[]
for bigram in pos_bigram:
    rand_neg=(bigram[0],random.choice(vocab))
    neg_bigram.append(rand_neg)
    rand_neg=(bigram[0],random.choice(vocab))
    neg_bigram.append(rand_neg) 
#df = pd.DataFrame(np.array(pos_bigram).reshape(-1,-1), columns = list("1"))
    


pos_df = pd.DataFrame(pos_bigram, columns=['first_word','second_word'])
pos_df['tag']="pos"

neg_df = pd.DataFrame(neg_bigram, columns=['first_word','second_word'])
neg_df['tag']="neg"


df = pos_df.append(neg_df, ignore_index=True)

count_vect = CountVectorizer(preprocessor=lambda x:x,
                                 tokenizer=lambda x:x)
X = count_vect.fit_transform(doc for doc in pos_bigram)


                    
                
               