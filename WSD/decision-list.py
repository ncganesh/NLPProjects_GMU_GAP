'''
WSD by Team GAP
Team Members: Pranav Krishna SJ,Alagappan A, Ganesh Nalluru
Date: 04/04/2019

Introduction
--------------
This program does word sense disambiguation for a given context,
the ambiguous word take into consideration is "line" which has two
meanings, "Product" and "phone". The task is to identify the sense of
the word "line" in the given context.

Reference: Yarowsky-Decision-List-1994(https://www.aclweb.org/anthology/P94-1013)

Features Used:
    One word before ambiguous word.
    One word after ambiguous word.
    Two words before ambiguous word.
    Two words after ambiguous word.
    One word before and after ambiguous word.
    Two words before and after ambiguous word.
    One part of speech tag before ambiguous word.
    One part of speech tag after ambiguous word.
    Along with these features some rules were written. If a particular word is in context
    then a particular sense is assigned to the word.
    
Accuracy:
    We recieved an accuracy of 93% on the test data.
    
Example
-------
Consider the below example,
The New York plan froze basic rates, offered no protection to Nynex against an economic
downturn that sharply cut demand and didn't offer flexible pricing.In contrast, the California economy is booming,
with 4.5% access <line> growth in the past year.

Here, the word inside angular bracket is ambiguous. Based on the neighbouring words the sense here is "phone".

Another example,
Culinova fresh entrees, launched in 1986 by Philip Morris Cos.'s General Foods Corp., hit similar distribution
problems.Last December, shortly after Philip Morris bought Kraft Inc., the struggling <line> was scrapped.

Here, the word "line" is surrounded with words relation to products. So, the sense here is "product"


Usage Instruction
-----------------
The Program consists of 2 python codes, "decision-list.py" and "scorer.py"

Windows
-------
1.Open cmd prompt, navigate to the location where the python codes are stored along with line-train.xml,line-test.xml
and line-answers.txt

2.Run the following command "python decision-list.py line-train.xml line-test.xml my-decision-list.txt >my-line-answers.txt",
this creates 2 text files, "my-decision-list.txt" consists of log-likelihood scores for each collocation type and the
predicted sense for a particular context. "my-line-answers.txt" is expected to have the instance id along with the
predicted sense which will be used by "scorer.py"

3.Run the following command "python scorer.py my-line-answers.txt line-answers.txt", this compares the result generated
by the program "decision-list.py" with the given key "line-answers.txt".

4.Based on the comparison a confusion matrix is generated along with the accuracy of the program.

Linux
-----
Follow the same steps as above, instead of command prompt use terminal to run the above commands.


Algorithm
---------
    1.Get train, test files from the user.
    2.Extract the "context" from "line-train.xml" document, along with its senseid.
    3.Identify the ambiguous word(<head>line</head>).
    4.Identify words surrounding the ambiguous word
        4.1. Sub divide into 'Bigram','Unigram','Plus2_Words','Minus2_Words','Plus1_Word','Minus1_Word',
        'Before_POS_Tag','After_POS_Tag'
        4.2 Based on which update the count value for both the senseids("Product","phone"). For example, count how many
        times a set of bigram words is associtaed with a senseid.
        4.3 Store it in different dataframes, namely "Bigram','Unigram','Plus2_Words'.......
    5.Finally, extract the "context" from "line-test.xml".
    6.Run a single context against all the created data frames and generate log likelihood for each data frame, find the
    max log likelihood value along with its associated dataframe.
    7.The selected dataframe is looked up. Based on the max count of the senseid, the sense is linked with the context.
    8.If log likelihood is same along all the data frames then a random choice is made between "product" and "phone"


'''

#Importing required packages
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
import nltk
import random
#from nltk.corpus import stopwords 
#from nltk.tokenize import RegexpTokenizer
import sys
#from nltk.corpus import stopwords
from collections import Counter


#Command Line Inputs
path=sys.argv[1]
tree=ET.parse(path)

path2=sys.argv[2]
tree1=ET.parse(path2)

decision_list=sys.argv[3]


random.seed(12345)
#Fetches the xml file and stores it in variable called tree
#Points to the root of the tree
root = tree.getroot()
line=root.find('lexelt')

#Returns two words before and after the ambigus word
def bigram(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace(",", "")
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    features=re.findall('(\w+ \w+)"?.?-? <head>\w{4,5}<\/head> (\w+ \w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)
#Returns k words before ambigus word
def kwordsBefore(xmlstr,k):
    features=[]
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace(",", "")
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(".", " ")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    cleanedWords = xmlstr.replace(",", "")
    bag_of_words=cleanedWords.split()
    
    if('<head>line</head>' in cleanedWords):
        index=bag_of_words.index('<head>line</head>')
    elif('<head>lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>lines</head>')
    elif('<head>Lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>Lines</head>')
    elif('<head>Line</head>' in bag_of_words):
        index=bag_of_words.index('<head>Line</head>')
        
    for i in range(1,k+1):
        features.append(bag_of_words[index-i])
    
    return(features)
    
#Returns k words after ambigus word
def kwordsAfter(xmlstr,k):
    xmlstr=str(xmlstr).lower()
    features=[]
    xmlstr = xmlstr.replace(",", "")
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(".", " ")
    cleanedWords = xmlstr.replace(",", "")
    bag_of_words=cleanedWords.split()
    
    if('<head>line</head>' in cleanedWords):
        index=bag_of_words.index('<head>line</head>')
    elif('<head>lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>lines</head>')
    elif('<head>Lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>Lines</head>')
    elif('<head>Line</head>' in bag_of_words):
        index=bag_of_words.index('<head>Line</head>')

    for i in range(1,k+1):
        if(len(bag_of_words)>index+i):
            features.append(bag_of_words[index+i])
        else:
            features.append('E O L')
    return(features)

#Returns a word before and after the ambigus word
def unigram(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    features=re.findall('(\w+)"?.?-? <head>\w{4,5}<\/head> (\w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)
    return(features)
#Returns a word before the ambigus word
def kminus1(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    features=re.findall('"?(\w+)"?.?-? <head>', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)
    return(features)
#Returns a word after the ambigus word
def kplus1(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    features=re.findall(r'</head> [,.;"]? ?(\w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)

#Tags POS for all the words and returns the tag of the words before and after the 
#ambiguous word
def tagger(xmlstr):
    #features=re.findall('</head> (\w+)', str(xmlstr), re.IGNORECASE)
    index=None
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    querywords=xmlstr
    #querywords = str(xmlstr).replace('</s>','').replace('<s>','').replace('</context>','').replace('\n','').replace('<context>','').replace('\\n\\n','').replace('\\n','')     
    wordsList=querywords.split()
    if('<head>line</head>' in wordsList):
        index=wordsList.index('<head>line</head>')
    elif('<head>lines</head>' in wordsList):
        index=wordsList.index('<head>lines</head>')
    textforTagging=querywords.replace('<head>','').replace('</head>','')
    store=nltk.pos_tag(textforTagging.split())  
    if(index):
        beforeTag=store[index-1][1]
        afterTag=store[index+1][1]
        return([beforeTag,afterTag])
    else:
        return(['NN','NN'])
        
        
beforeWord=[]
afterWord=[]
unigramFeature=[]
bigramFeature=[]
beforeTag=[]
afterTag=[]
senses=[]
kminus2Feature=[]
kplus2Feature=[]
instanceId=[]
productList=''
senseList=''
#Iterates through all the instances, finds and extracts features and then appends
#them to appropriate lists
for val in line:
    #Stores the instance ID
    instanceId.append((val.attrib)['id'])
    #Iterates through all the tags under instance tag
    for context in val:
        #If the tag has sense id the that is stored in the senseId variable
        if((context.attrib)!={}):
            senseId=(context.attrib)['senseid']
            senses.append(senseId)
        #If the tag contains the string and the ambiguous word, fowwing are performed
        else:
            #Converts the tags into strings
            xmlstr = ET.tostring(context)
            if(senseId=='product'):
                productList+=(str(xmlstr).lower())
            else:
                senseList+=(str(xmlstr).lower())
            #Gets the word before the ambigus word
            beforeWord.append(((kminus1(xmlstr))[0],senseId))
            #Gets the word after the ambigus word
            afterWord.append(((kplus1(xmlstr))[0],senseId))
            #Gets a word before and after the ambigus word
            unigramString=' '.join((unigram(xmlstr))[0])
            unigramFeature.append((unigramString,senseId))
            #Gets two words before and after the ambigus word
            bigramString=' '.join((bigram(xmlstr))[0])
            bigramFeature.append((bigramString,senseId))
            # Gets the tag before and after the ambiguous word
            tags=tagger(xmlstr)
            beforeTag.append((tags[0],senseId))
            afterTag.append((tags[1],senseId))
            
            # Gets 2 words before and after the ambiguous word
            kminus2Feature.append((' '.join(kwordsBefore(xmlstr,2)),senseId))
            kplus2Feature.append((' '.join(kwordsAfter(xmlstr,2)),senseId))

#The following code finds the words and its frequency for words which are exclusively in sentences having a particular
#sense. Those words can be used to write rules. The code is commented out because those words did not help in improving 
#accuracy
'''
productList=str(productList.lower())
productList = productList.replace("<s>","")
productList = productList.replace("</s>", "")
productList = productList.replace(r'\n', "")
productList = productList.replace("<context>", "")
productList = productList.replace("</context>", "")

senseList=str(senseList.lower())
senseList = senseList.replace("<s>","")
senseList = senseList.replace("</s>", "")
senseList = senseList.replace(r'\n', "")
senseList = senseList.replace("<context>", "")
senseList = senseList.replace("</context>", "")

stopWords=set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\w+')

cleanedWordsProductList=tokenizer.tokenize(productList)
cleanedWordsProductList=[word for word in cleanedWordsProductList if not word in stopWords]

cleanedWordsPhoneList=tokenizer.tokenize(senseList)
cleanedWordsPhoneList=[word for word in cleanedWordsPhoneList if not word in stopWords]

intersection = Counter(cleanedWordsProductList) & Counter(cleanedWordsPhoneList)
ProductListCounts=Counter(cleanedWordsProductList)-intersection
PhoneListCounts=Counter(cleanedWordsPhoneList)-intersection
PhoneListCounts.most_common(305)

#list(nltk.FreqDist(cleanedWords))
'''
#Finds unique elements in the list in order
def setz(sequence):
    sequenceList=[]
    for i in sequence:
        if(i[0] not in sequenceList):
            sequenceList.append(i[0])
    return(sequenceList)


#Initializes a dataframe with ones. 1 is used instead of 0 to perform smoothing.
def initialize(sequence1,sequence2):
    return(np.ones((len(sequence1), len(sequence2))))
 
#The following function builds a dataframe with features and its sense count.
def dfBuilder(featureSequence):
    uniqueList=setz(featureSequence)
    initializeSet = initialize((uniqueList), set(senses))
    initializedFeature=pd.DataFrame(initializeSet, index=uniqueList, columns=set(senses))
    for i in featureSequence:
        initializedFeature.at[i[0], i[1]]=initializedFeature.at[i[0], i[1]] + 1
    return(initializedFeature)

#Building a dataframe for each feature
oneWordBefore=dfBuilder(beforeWord)
oneWordAfter=dfBuilder(afterWord)
unigramwords=dfBuilder(unigramFeature)
bigramwords=dfBuilder(bigramFeature) 
kminus2words=dfBuilder(kminus2Feature)
kplus2words=dfBuilder(kplus2Feature)
beforeTagDf=dfBuilder(beforeTag)
afterTagDf=dfBuilder(afterTag)


# neighbouring words and flag is passed as an argument to prob_finder
def prob_finder(temp_list,flag):
    # if the word is "not end of line " enters condition
    if (temp_list != 'E O L'):
        # based on the passed flag value enters the conditions
        if(flag==0):
            # merges the words
            mer = temp_list[0][0] + " " + temp_list[0][1]
            # checks if the word exist in the collocation dataframe
            if(mer in bigramwords.index):
                # gets the associated sense count value
                product = bigramwords.at[mer, 'product']
                phone = bigramwords.at[mer, 'phone']
            # other wise counts of sense is set to 0
            else:
                product=0
                phone=0

        elif(flag==1):
            mer = temp_list[0][0] + " " + temp_list[0][1]
            if (mer in unigramwords.index):
                product = unigramwords.at[mer, 'product']
                phone = unigramwords.at[mer, 'phone']

            else:
                product = 0
                phone = 0

        elif (flag == 2):

            mer = temp_list[0] + " " + temp_list[1]
            if (mer in kplus2words.index):
                product = kplus2words.at[mer, 'product']
                phone = kplus2words.at[mer, 'phone']

            else:
                product = 0
                phone = 0

        elif (flag == 3):
            mer = temp_list[0] + " " + temp_list[1]
            if(mer in kminus2words.index):
                product = kminus2words.at[mer, 'product']
                phone = kminus2words.at[mer, 'phone']

            else:
                product = 0
                phone = 0

        elif (flag == 4):
            if(temp_list[0] in oneWordAfter.index):
                product = oneWordAfter.at[temp_list[0], 'product']
                phone = oneWordAfter.at[temp_list[0], 'phone']

            else:
                product = 0
                phone = 0

        elif (flag == 5):
            if(temp_list[0] in oneWordBefore.index):
                product = oneWordBefore.at[temp_list[0], 'product']
                phone = oneWordBefore.at[temp_list[0], 'phone']

            else:
                product = 0
                phone = 0
            
        elif (flag == 6):
            if(temp_list[0] in beforeTagDf.index):
                product = beforeTagDf.at[temp_list[0], 'product']
                phone = beforeTagDf.at[temp_list[0], 'phone']

            else:
                product = 0
                phone = 0
            
        elif (flag == 7):
            if(temp_list[0] in afterTagDf.index):
                product = afterTagDf.at[temp_list[0], 'product']
                phone = afterTagDf.at[temp_list[0], 'phone']

            else:
                product = 0
                phone =    0

        # Checks if both product and phone have the same count value
        # if yes then a random sense is returned
        if (product == phone):
            prob = max(product, phone)
            wsd=random.choice(["product","phone"])
        # Otherwise the log likelhood is calculated
        else:
            prob = np.log(product / phone)

            # Based on the count value the sense is returned
            if(product>phone):
                wsd="product";
            else:
                wsd="phone"
    # if the word is "end of line" then probability is set to 0
    # A random choice is returned
    else:
        prob = 0
        wsd = random.choice(["product", "phone"])


    return abs(prob),wsd


#Points to the root of the tree1
root = tree1.getroot()
line1=root.find('lexelt')

# The line-test.xml is extracted of context and is cleaned for further precessing.
content=[]
instanceId=[]
for val in line1:
    instanceId.append((val.attrib)['id'])
    for context in val:
        temp=str(ET.tostring(context))
        temp = temp.replace("<s>","")
        temp = temp.replace("</s>", "")
        temp = temp.replace(r'\n', "")
        temp = temp.replace("<context>", "")
        temp = temp.replace("</context>", "")
        content.append(temp)



count=0
bi_prob=[]
uni_prob=[]
minus2_prob=[]
plus2_prob=[]
minus1_prob=[]
plus1_prob=[]
beforeTag_prob=[]
afterTag_prob=[]
final=[]
collectionList=[]
x=-1
# Each context runs through a loop
# The neighbouring word is extracted based on the collocation(bigram, unigram...) and stored in a variable
# Setting the flag value
# passing the stored variable along with flag value to prob finder
# return probability and respective sense
# appending the returned probability in a list
for i in content:
    jj=None
    x+=1

    a = bigram(i)
    flag=0
    prob,aa=prob_finder(a,flag)
    bi_prob.append(prob)

    b = unigram(i)
    flag=1
    prob,bb = prob_finder(b,flag)
    uni_prob.append(prob)

    c = kwordsAfter(i, 2)
    flag = 2
    prob,cc = prob_finder(c, flag)
    plus2_prob.append(prob)

    d =kwordsBefore(i,2)
    flag = 3
    prob,dd = prob_finder(d, flag)
    minus2_prob.append(prob)

    e = kplus1(i)
    flag = 4 
    prob,ee = prob_finder(e, flag)
    plus1_prob.append(prob)

    f = kminus1(i)
    flag = 5
    prob,ff = prob_finder(f, flag)
    minus1_prob.append(prob)
    
    tags=tagger(i)
    g=[tags[0]]
    flag=6
    prob,gg = prob_finder(g, flag)
    beforeTag_prob.append(prob)
    
    h=[tags[1]]
    flag=7
    prob,hh = prob_finder(h, flag)
    afterTag_prob.append(prob)
    
    tokenizedString=nltk.word_tokenize(i.lower())
    productSet=['company']
    phoneSet=['telephone']
    

    # few rules to improve the accuracy, the obvious neighbouring words are associated with their senses
    if('consumer' in tokenizedString):
        jj='product'
    
    elif('phone' in tokenizedString):
        jj='phone'
       
    elif('telephone' in tokenizedString):
        jj='phone'
        
    elif('call' in tokenizedString):
        jj='phone'

    elif('calls' in tokenizedString):
        jj='phone'

    # storing all the probabilities in list
    collection=[bi_prob[x],uni_prob[x],plus2_prob[x],minus2_prob[x],plus1_prob[x],minus1_prob[x],beforeTag_prob[x],afterTag_prob[x]]
    # appending all the probabilities in list
    collectionList.append(collection)
    # finding the max probability out of all the collocation
    reorder=np.argmax([bi_prob[x],uni_prob[x],plus2_prob[x],minus2_prob[x],plus1_prob[x],minus1_prob[x],beforeTag_prob[x],afterTag_prob[x]])

    # the selected probability assigns the sense and appends the final list
    if(jj):
        final.append(jj)        
    elif(reorder==7):
        final.append(hh)
    elif(reorder==6):
        final.append(gg)
    elif(reorder==5):
        final.append(ff)
    elif(reorder==4):
        final.append(ee)
    elif (reorder == 3):
        final.append(dd)
    elif (reorder == 2):
        final.append(cc)
    elif (reorder == 1):
        final.append(bb)
    elif (reorder == 0):
        final.append(aa)

# Creating a data frame to store all the log likelihood and associated senses
col=['Bigram','Unigram','Plus2_Words','Minus2_Words','Plus1_Word','Minus1_Word','Before_POS_Tag','After_POS_Tag']
decision_list_df=pd.DataFrame()

for i in range(0,len(collectionList)):
    for j in range(0,len(col)):
        decision_list_df.loc[i, col[j]]=collectionList[i][j]

decision_list_df['Sense']=final

# Storing it in a text file
with open('my-decision-list.txt', 'w') as f:
    f.write(decision_list_df.to_string())

# Creating a text file with instance id and associated senses in "line-answers.txt" format
my_list=""
for i in range(0,len(instanceId)):
    temp='<answer instance="'+instanceId[i]+'" senseid="'+final[i]+'"/>'
    my_list=my_list+temp+'\n'

print(my_list)
