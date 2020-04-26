# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 01:53:14 2019

@author: Pranav Krishna
"""

import os
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
import re

#rootdir= r'D:\\bin\\AIT-726\\Assignemnts\\Sentiment_Classification_Data\\train\pos\\' 
#use '\\' in a normal string if you mean to make it be a '\'   
#use '\ ' in a normal string if you mean to make it be a ' '   


ps = PorterStemmer()

def collect_vocab(line,vocab):
    vocab.extend(word_tokenize(line))
    #let the function return, not only print, to get the value for use as below 

def normalize_case(s):
    if(not s.isupper()):
        return s.lower()
    else:
        return s
    
def remove_tags(s):
    s=re.sub(r'<.>', '', s)
    return s


def count_words(rootdir):    
    vocab=[]
    vocab_case=[]
    stemmed_vocab=[]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            f=open(rootdir+file,'r',encoding='utf-8') #use the absolute URL of the file
            lines = f.readlines()
            for line in lines:
                vocab.extend(word_tokenize(line))
            f.close()
    for word in vocab:
        word=normalize_case(word)
        word=remove_tags(word)
        if(word!=''):
            vocab_case.append(word)
    for word in vocab_case:
        stemmed_vocab.append(ps.stem(word))    
    count=dict(collections.Counter(vocab_case))
    return count

def likelyhood(word_count,class_count,total_count):
    likelyhood_dict={}
    for key in word_count:
        likelyhood_dict[key]=(word_count[key])/(class_count+total_count)
    return likelyhood_dict
    
pos_count={}
neg_count={}
total_pos_words=0
total_neg_words=0
def nb():
    global pos_count
    global neg_count
    global total_neg_words
    rootdir= r'D:\\bin\\AIT-726\\Assignemnts\\Sentiment_Classification_Data\\train\pos\\' 
    pos_count=count_words(rootdir)
    global total_pos_words
    total_pos_words=len(pos_count)
    
    rootdir= r'D:\\bin\\AIT-726\\Assignemnts\\Sentiment_Classification_Data\\train\neg\\' 
    neg_count=count_words(rootdir)
    total_neg_words=len(neg_count)
    
    #pos_likeylyhood=likelyhood(pos_count,total_pos_words,total_pos_words+total_neg_words)
   # neg_likeylyhood=likelyhood(neg_count,total_neg_words,total_pos_words+total_neg_words)
    #print(pos_likeylyhood['of'])

nb()

test_files=[]
def get_test(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            document=[]
            f=open(rootdir+file,'r',encoding='utf-8') #use the absolute URL of the file
            lines = f.readlines()
            for line in lines:
                document.append(word_tokenize(line))
            test_files.extend(document)
            f.close()
rootdir= r'D:\\bin\\AIT-726\\Assignemnts\\Sentiment_Classification_Data\\test\pos\\'
get_test(rootdir)

doc_classification={}
i=1
for doc in test_files:
    pos_word_likelyhood=1
    neg_word_likelyhood=1
    for word in doc:
        pos_word_likelyhood=pos_word_likelyhood*((pos_count.get(word,0)+1))
        neg_word_likelyhood=neg_word_likelyhood*((neg_count.get(word,0)+1))
    if(pos_word_likelyhood > neg_word_likelyhood):
        doc_classification[i]='pos'
    elif(pos_word_likelyhood < neg_word_likelyhood):
        doc_classification[i]='neg'
        #print(pos_word_likelyhood)
        #print(neg_word_likelyhood)
    i=i+1
    
    
rootdir= r'D:\\bin\\AIT-726\\Assignemnts\\Sentiment_Classification_Data\\test\neg\\'
get_test(rootdir)

doc_classification={}
i=1
for doc in test_files:
    pos_word_likelyhood=1
    neg_word_likelyhood=1
    for word in doc:
        pos_word_likelyhood=pos_word_likelyhood*((pos_count.get(word,0)+1)/2*)
        neg_word_likelyhood=neg_word_likelyhood*((neg_count.get(word,0)+1))
    if(pos_word_likelyhood > neg_word_likelyhood):
        doc_classification[i]='pos'
    elif(pos_word_likelyhood < neg_word_likelyhood):
        doc_classification[i]='neg'
        #print(pos_word_likelyhood)
        #print(neg_word_likelyhood)
    i=i+1

