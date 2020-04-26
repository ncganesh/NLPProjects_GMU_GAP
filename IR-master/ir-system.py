#Importing necessary libraries
import numpy as np
import sys


content=sys.argv[1]
query=sys.argv[2]


#Preprocessing docfile to extract index, title and body seperately
content=open(content)
content=content.read()
content = content.replace("\n", "")
content=content.replace(".I"," .I")
content=content.replace(".T"," .T ")
content=content.replace(".W"," .W ")
content=content.replace("-"," - ")


#temp=word_tokenize(content)
#Removing stop words
con_split=content.split()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

tempo= []
for w in con_split:
    if w not in stop_words:
        tempo.append(w)

from nltk.stem import PorterStemmer

#Performs word stemming
ps = PorterStemmer()
temp=[]
for w in tempo:
    if '.A' not in w:
        temp.append(ps.stem(w))
    else:
        temp.append(w)

#Appends indexes, title and body of texts to appropriate lists
cnt=0
ids=[]
title=[]
body=[]
flag=-1
flag1=-1
for i in range(0,len(temp)):
    if '.I'in temp[i]:
        flag=0
    elif(flag==0):

        ids.append(temp[i])
        flag=1
    elif(flag==1):
        if('.T' in temp[i]):
            title.append(temp[i])
            flag1=0
        elif(flag1==0 and '.A' not in temp[i]):
            title.append(temp[i])
        elif('.A' in temp[i]):
            flag=2
    elif temp[i]=='.W' and flag==2:
        flag=3
        body.append("**")
    elif(flag==3):
        body.append(temp[i])


del title[0]
title_split=' '
title_split=title_split.join(title)
title_split=title_split.split('.T')


del body[0]
body_split=' '
body_split=body_split.join(body)
body_split_docs=body_split.split('**')

#Store docs as key value pairs with ids as keys and the document and their title as 
#values
docs={}
for i in range(0,len(ids)):
    docs[i+1]=[title_split[i],body_split_docs[i]]

    
#query = r"D:\bin\AIT-690\Assignments\IR\cran.qry"

#query = r"C:\Users\alaga\Desktop\sem 2\AIT690\IR1\cran.qry"

#Preprocessing queries to extract indexes and body of texts 
query=open(query)
query=query.read()
query = query.replace("\n", "")
query=query.replace(".I"," .I")
query=query.replace(".W"," .W ")
query=query.replace("-"," - ")

que_split=query.split()
tempo= []
for w in que_split:
    if w not in stop_words:
        tempo.append(w)


ps = PorterStemmer()
#Performs stemming
temp=[]
for w in tempo:
    if '.A' not in w:
        temp.append(ps.stem(w))
    else:
        temp.append(w)
        
#Appends indexes, title and body of texts to appropriate lists for queries
cnt=0
ids=[]
title=[]
body=[]
flag=-1
flag1=-1
for i in range(0,len(temp)):
    if '.I'in temp[i]:
        flag=0
    elif(flag==0):
        ids.append(temp[i])
        flag=1
    elif temp[i]=='.W' and flag==1:
        flag=3
        body.append("**")
    elif(flag==3):
        body.append(temp[i])

del body[0]
body_split=' '
body_split=body_split.join(body)
body_split=body_split.split('**')

#Function to calculate tf
#@Params-term and document
#retuens tf
def termFrequency(term, document):
    normalizeDocument = document.lower().split()
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))

#Function to calculate idf
#@Params-term and all documents
#retuens idf
def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    for doc in allDocuments:
        if term.lower() in allDocuments[doc][0].lower().split():
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1

    if numDocumentsWithThisTerm > 0:
        return 1.0 + np.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0

def square(list):
    return map(lambda x: x ** 2, list)

#Function to implement jaccard similarity
#@params-query and document
#@Returns Jaccard similarity score
#def jaccard_similarity(list1, list2):
    #intersection = len(set(list1).intersection(list2))
    #union = (len(list1) + len(list2)) - intersection
    #return float(intersection / (union+1))

#The function finds the cosine similarity between queries and documnets
#@params-query and document
#returns-cosine similarity score
def cosineSimilarity(query, doc):
    up = float(np.dot(query, doc))
    modQuery = np.sqrt(sum(square(query)))
    docQuery = np.sqrt(sum(square(doc)))
    down = float(modQuery * docQuery)
    if(down==0):
        down=1
    return (up / down)


# content_title
# indexing the idfs
indexing_idf_word={}
len(ids)
for i in range(1,len(docs)+1):
    x = docs[i][1].split()
    for j in x:
        if j not in indexing_idf_word:
            indexing_idf_word[j]=inverseDocumentFrequency(j, docs)

doc_query_tf=[]
doc_query_idf=[]
#total_jaccard_similarityScore=[]
#For each query
for i in body_split:
    #jaccard_similarityScore={}
    doc_query_word_tf = []
    doc_query_word_idf = []
    #For each document
    for k in range(1, len(docs)+1):
        doc_query_word_doc_tf = []
        doc_query_word_doc_idf = []
        #jaccard_similarityScore[k]=jaccard_similarity(i.split(), body_split_docs[k-1].split())
        #For each word in query
        for j in i.split():
            #If the is in the current document
            if j in docs[k][1].split():
                #Find and append the term frequency of the word in document to the list
                doc_query_word_doc_tf.append(termFrequency(j, docs[k][1]))
            else:
                doc_query_word_doc_tf.append(0)
                #If the documnt has an Idf
            if j in indexing_idf_word:
                #Find and append the idf to the list
                doc_query_word_doc_idf.append(indexing_idf_word[j])
            else:
                doc_query_word_doc_idf.append(0)

        #Store all the term frequencies of the document in a list 
        doc_query_word_tf.append(doc_query_word_doc_tf)
        #Store all the inverse document frequencies of the document in a list
        doc_query_word_idf.append(doc_query_word_doc_idf)
    #Sorts the itmes based on jaccard Similarity Score
    #sorted_x = sorted(jaccard_similarityScore.items(), key=lambda kv: kv[1],reverse=True)
    #total_jaccard_similarityScore.append(sorted_x)
    
    #Stores all the term frequencies of the all the documents in a list 
    doc_query_tf.append(doc_query_word_tf)
    #Stores all the inverse document frequencies of all the document in a list
    doc_query_idf.append(doc_query_word_idf)


#The follwing code multiplies the term frequency with the inverse document frequency
complete_doc_query_tf_idf=[]
for i in range(0,len(body_split)):
    a=doc_query_tf[i]
    b=doc_query_idf[i]
    complete_doc_query_tf_idf.append(np.multiply(a,b))

#finds term frequency of queries
query_idf=[]
query_tf=[]
for i in body_split:
    query_word_tf=[]
    query_word_idf = []
    for j in i.split():
        query_word_tf.append(termFrequency(j, i))
        if j in indexing_idf_word:
            query_word_idf.append(indexing_idf_word[j])
        else:
            query_word_idf.append(0)

    query_idf.append(query_word_idf)
    query_tf.append(query_word_tf)

#Finds the final tf-idf for queries
complete_query_tf_idf=[]
for i in range(0,len(body_split)):
    a=query_tf[i]
    b=query_idf[i]
    complete_query_tf_idf.append(np.multiply(a,b))


#For each query finds the cosine similarity between that query
#and all the documents and appends that to the list
complete_list=[]
for i in range(0,len(body_split)):
    list = []
    for k in range(0, len(docs)):
        x=cosineSimilarity(complete_query_tf_idf[i],complete_doc_query_tf_idf[i][k])
        list.append(x)
    complete_list.append(list)

#Sorts the cosine simalrity and finds the ranking of all the documents for each query
final_list = []
for i in range(0, len(complete_list)):
    temp = np.argsort(complete_list[i])
    temp = temp[::-1]
    sub_final_list = []
    for j in range(0, len(temp)):
        if complete_list[i][temp[j]] >= 0.55:
            sub_final_list.append(temp[j] + 1)
    final_list.append(sub_final_list)

output = []
for i in range(0, len(final_list)):
    for j in final_list[i]:
        output.append(str(i + 1) + ' ' + str(j))

with open('cran-output.txt', 'w') as f:
    for item in output:
        f.write("%s\n" % item)

#with open('jaccard.txt', 'w') as f:
    #for index, item in enumerate(total_jaccard_similarityScore):
        #for i in item:
            #f.write(str(index + 1) + " " + str(i[0]) + "\n")

