Information retrieval system by team GAP.
--------------


Introduction
This program retrives relavant documents for the given query by calculating tf-idf
vectors for document and queries and by calculating similarity score between them.

Example
--------
Query:

    .I 001
    .W
    what similarity laws must be obeyed when constructing aeroelastic models
    of heated high speed aircraft .
    
Documents:

    .I 184
    .T
    scale models for thermo-aeroelastic research .
    .A
    molyneux,w.g.
    .B
    rae tn.struct.294, 1961.
    .W
    scale models for thermo-aeroelastic research .
    An investigation is made of the
    parameters to be satisfied for
    thermo-aeroelastic similarity .  it is concluded
    that complete similarity obtains
    only when aircraft and model are identical
    in all respects, including size.
    
    .I 13
    .T
    similarity laws for stressing heated wings .
    .A
    tsien,h.s.
    .B
    j. ae. scs. 20, 1953, 1.
    .W
    similarity laws for stressing heated wings .
      it will be shown that the differential equations for a heated
    plate with large temperature gradient and for a similar plate at
    constant temperature can be made the same by a proper
    modification of the thickness and the loading for the isothermal plate .
    this fact leads to the result that the stresses in the heated plate
    can be calculated from measured strains on the unheated plate by
    a series of relations, called the /similarity laws .
    
Output:
-------
    Query Number Document Order
        1           13
        1           184
    
(Document length has been shortened for brevity)

Usage Instructions
------------------
Windows
-------
1.Open cmd prompt, navigate to the location where the python codes are stored along with query
and document files

2.Run the following command "python ir-system.py.py  cran.all.1400  cran.qry",
this creates a text file, "cran-output.txt" consists of query indices and relevant documet indices

3.Run the following command "python precision_recall.py cran-output.txt cranqrel,", this compares the result generated
by the program with the given key .

4.Based on the comparison a file with precision and recall numbers is created with name
mylogfile.txt.
Linux
-----
Follow the same steps as above, instead of command prompt use terminal to run the above commands.
Algorithm:
    Extract title, index  and body from the documets
    Preprocess the documents to remove stop words
    Find TF-IDF of each word in the title of the document
    Find TF of each word in the query
    Find similarity score between query and document
    Store the results with the descending order of similarity score
    

Stop Words,punctuation and stemming.
-----------

Removing stop words and stemming helped us increase the mean average precision wheras removing 
punctuations surprisingly reduced the mean average precision.

Improvement over the model

We observed that usage of idf affected the model by not taking advantage of context. So we tried
a model which finds the relevancy between documents by using jaccard similarity which does not
consider document frequency.



Jaccard Similarity
-------------------

Jaccard similarity finds a similarity score based on intersecting words between query and the
document. Unlike cosine similarity, it does not take any type of vectors into account. 

Jaccard similarity is calculated by finding the count of unique words in query which are 
intersecting with the document divided by the total number ofunique words which are not 
intersecting with the document.

Jaccard similarity did not give good results as the cosine similarity. The Mean Average Precision
by jaccard similarity is just 0.11.(The code for finding jaccard similarity is commented out duw to 
low scores)

Error analysis for jaccard model
---------------------------------

We can understand why our model is failing when we look at the errors. 

Index of the query and number of irrelavant documents returned for that query using both the models
are given below.(Sorted by queries which retruned most number of irrelavant documets)

Cosine Model 
[(122, 1385),  
 (152, 589),
 (70, 171),
 (150, 120),
 (64, 116),
 (30, 99),
 (181, 84),
 (182, 83),
 (219, 77),
 (133, 71)]

Jaccard Model
[(22, 1398),
 (31, 1398),
 (93, 1398),
 (119, 1398),
 (142, 1398),
 (216, 1398),
 (4, 1397),
 (14, 1397),
 (15, 1397),
 (17, 1397)]

 We can observe that the jaccard model gives lot of irrelavnt documents even after setting up a 
 significant threshold. This was not the case in the model which finds cosine similarity. In cosine 
 model changing the threshold helped us decrease the count of irrelavant documets. So this did not solve 
 the purpose we intended to solve.

'''
