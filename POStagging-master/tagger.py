# Import packages
import nltk
import sys


# arguments from command line are stored as train and test
train=sys.argv[1]
test=sys.argv[2]

# opening argument 1(pos-train.txt) and storing it as trainset
trainset = open(train)

# reading the train set as tagged_trainset
tagged_trainset = trainset.read()

# removing "[,],\n" which are not needed
tagged_trainset = tagged_trainset.replace("[", "")
tagged_trainset = tagged_trainset.replace("]", "")
tagged_trainset = tagged_trainset.replace("\n", "")

# extacting tags from the give tagged trainset
only_tags = []
taggedList = []
for line in tagged_trainset.split():
    # using the str2tuple nltk function to split the word and tags
    tagTuples = nltk.tag.str2tuple(line)
    # handling the anomaly of multiple tags for single word
    if (tagTuples[1] != None):
        if ("|" in list(tagTuples[1])):
            tagTuples = (tagTuples[0], tagTuples[1].split('|')[0])
        # split words and tags are stored in a list
        taggedList.append(tagTuples)
        # only the tags are stored in the only_tags list based on their order
        only_tags.append(tagTuples[1])

# each word has multiple tags, taggedDict holds all the tags that a corresponding word holds
taggedDict={}
for x,y in taggedList:
    if(x in taggedDict.keys()):
        taggedDict[x].append(y)
    else:
        taggedDict[x]=[y]

# from the only_tags list,
# each tag is counted and stored in single_tag_count and combination of two tags are counted and stored in double_tag_count
double_tag_count={}
single_tag_count = {}
for i in range(0,len(only_tags)):
    flag_single = 0
    if (only_tags[i] in single_tag_count.keys() and flag_single == 0):
        single_tag_count[only_tags[i]] = single_tag_count[only_tags[i]] + 1
        flag_single = 1
    elif(flag_single == 0):
        single_tag_count[only_tags[i]] = 1
    if(i<len(only_tags)-1):
        mer = only_tags[i] + ' ' + only_tags[i + 1]
        if mer in double_tag_count.keys():
            double_tag_count[mer] = double_tag_count[mer] + 1
        else:
            double_tag_count[mer] = 1

# opening argument 2(pos-test.txt) and storing it as testset
testset = open(test)
#reading the testset as untagged_testset
untagged_testset = testset.read()

# TEXT cleaning, removing "[,],\n"
untagged_testset = untagged_testset.replace("[", "")
untagged_testset = untagged_testset.replace("]", "")
untagged_testset = untagged_testset.replace("\n", "")

# function to find unique pairs in a list
def setz(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

# function to calculate the probability of tag based on the prev tag and the current word
# returns the tag of the particular word based on the calculated probability
# Parameters: Previous tag and current word
def tagged_word(prev,curr):
    max_prob = []
    # if the word exists in the taggedDict, enters..
    if curr in taggedDict:
        # finds the length of the tags associated with words
        n = len(taggedDict[bigram[i]])
        # gets the unique list of tags for the particular word
        unique = list(setz(taggedDict[bigram[i]]))
        for u in unique:
            # counts the occurrence of single tag
            count_val = taggedDict[bigram[i]].count(u)
            # probability is count of a tag for the word by total number of occurrence of the selected tag
            prob1 = count_val / single_tag_count[u]
            # previous tag and the current tags are merged
            mer = prev + ' ' + u
            # check if merged tag is in the double_tag_count dictionary
            if mer in double_tag_count.keys():
                # probability is count of merged tags by total number if occurrence of the selected tag
                prob2 = double_tag_count[mer] / single_tag_count[prev]
                total_prob = prob1 * prob2
                # a list is maintained to store the probabilities
                max_prob.append(total_prob)
                # the max probability is identified from the list
                # the index value of the max prob is used to identify the tag that actually fits the word
                typeVal = unique[max_prob.index(max(max_prob))]

            # if the merged word does not exist then a unigram model is used
            # based on max count the tag is assigned
            elif bigram[i] in taggedDict:
                unique = list(setz(taggedDict[bigram[i]]))
                max_val = taggedDict[bigram[i]].count(unique[0])
                typeVal = unique[0]
                for u in unique:
                    n = taggedDict[bigram[i]].count(u)
                    if (n > max_val):
                        max_val = n
                        typeVal = u
            
        # the current tag is stored in prev for future reference
        prev = typeVal
        # returns the prev tag and the current tag
        return prev,typeVal


# basic structure

# the untagged test set is stored in bigram
bigram=untagged_testset
tagged_test_string=[]
tagged=''
mer=''
# bigram is split and stored in a list
bigram=bigram.split()
for i in range(0,len(bigram)):
    max_prob=[]
    # enters if the word exists in taggedDict
    if bigram[i] in taggedDict:

        # if only one tag is associated to a word then assign it
        if len(setz(taggedDict[bigram[i]]))==1:
            typeVal=taggedDict[bigram[i]][0]

        # if the word is a starting word
        elif i==0:
            # previous tag is to be '.'
            prev='.'
            # current word is stored in curr
            curr=bigram[i]
            # the previous tag and the current tag is passed as an argument to tagged_word function
            prev,typeVal=tagged_word(prev,curr)

        # otherwise
        else:
            curr = bigram[i]
            prev, typeVal = tagged_word(prev, curr)

    # rules for unseen word
    else:
        
        if bigram[i][0].isdigit()==True:
            typeVal = "CD"
        elif bigram[i][-1]=='s':
            typeVal = "NNS"
        elif bigram[i][-2:]=='ed':
            typeVal = "VBN"
        elif bigram[i][-4:] == 'able':
            typeVal = "JJ"
        elif bigram[i][0].isupper()==True:
            typeVal = "NNP"
        elif bigram[i][-3:] == 'ing':
            typeVal = "VBG"
        else: 
            typeVal = "NN"
        prev=typeVal

    # appending the word and tag
    tagged=bigram[i]+'/'+typeVal
    # storing the tag in order, in a list
    tagged_test_string.append(tagged)


# Set of rules after analyzing confusion matrix
for i in range(0,(len(tagged_test_string)-2)):
    currTag=nltk.tag.str2tuple(tagged_test_string[i])[1]
    nextTag=nltk.tag.str2tuple(tagged_test_string[i+1])[1]
    if(i>0):
        prevTag=nltk.tag.str2tuple(tagged_test_string[i-1])[1]
    currWord=nltk.tag.str2tuple(tagged_test_string[i])[0]
    
    # If the word is "a" and if the previous tags for the word does not denote
    # end of sentences, then that word is a determiner.
    
    if(currWord == 'a' and prevTag not in [',','.',':']):
        typeVal='DT'
        tagged=bigram[i]+'/'+typeVal
        tagged_test_string[i]=tagged
    
    # If the word is currently tagged as particle and if the word is not after different
    # forms of verb then the word should be tagged as preposition or  subordinating conjunction.
    
    elif(currTag == 'RP' and prevTag not in ["VB","VBD","VBG","VBN","VBZ","VBP"]):
        typeVal='IN'
        tagged=bigram[i]+'/'+typeVal
        tagged_test_string[i]=tagged
    # If the word is currently tagged as verb and if the previous tag is determiner,
    # then the word should be tagged as noun.
               
    elif(currTag == 'VB' and prevTag == "DT"):
        typeVal='NN'
        tagged=bigram[i]+'/'+typeVal
        tagged_test_string[i]=tagged
    
    # If the word is currently tagged as Wh-determiner and if the previous tag is not
    # a noun then the word should be a preposition or  subordinating conjunction.
          
    elif(currTag == 'WDT' and prevTag not in ["NN","NNS",',']):
        typeVal='IN'
        tagged=bigram[i]+'/'+typeVal
        tagged_test_string[i]=tagged
 
# writing the tagged_test_string to a text file
with open(r'pos-test-with-tags.txt', 'w') as f:
    for item in tagged_test_string:
        f.write("%s\n" % item)


print("pos-test-with-tags.txt")



