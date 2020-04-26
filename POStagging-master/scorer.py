'''
Team GAP
The scorer finds the accuracy and the confusion matrix of predicted and actual values. 
'''

# importing the packages
import sys
import nltk
import numpy as np
import pandas as pd


# accuracy test

predicted=sys.argv[1]
key=sys.argv[2]

# function to extract tags out of a given list
# Parameter:tagged_text-predicted tag from tagger.py
def extract_tags(tagged_text):
    only_tags = []
    for line in tagged_text:
        tagTuples = nltk.tag.str2tuple(line)
        if "|" in list(tagTuples[1]):
            tagTuples=(tagTuples[0],tagTuples[1].split('|')[0])
        only_tags.append(tagTuples[1])
    return only_tags

# finding unique elements in a ordered fashion
# parameter:list with non-unique values.
def setz(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


# opening argument 1(pos-test-with-tags.txt) and storing it as predicted
predicted=open(predicted)
# reading the predicted as predicted
predicted=predicted.read()


# opening argument 2(pos-test-key.txt) and storing it as tagged_testset_key
tagged_testset_key= open(key)
# reading the tagged_testset_key as tagged_test_set_key
tagged_testset_key=tagged_testset_key.read()

tagged_testset_key = tagged_testset_key .replace("[", "")
tagged_testset_key = tagged_testset_key .replace("]", "")
tagged_testset_key= tagged_testset_key .replace("\n", "")

# both predicted and actual pass their strings to extract_tags function to extract the tags
predicted=extract_tags(predicted.split())
actual=extract_tags(tagged_testset_key.split())

tags=setz(actual)

# Initialize confusion matrix with zeroes.
A = np.zeros((len(tags), len(tags)))
confusion_matrix = pd.DataFrame(A, index=tags, columns=tags)

# Builds the confusion matrix by incrementing values based on actual and predicted tags
cnt=0
for i in range(0,len(predicted)):
    if predicted[i]==actual[i]:
        cnt+=1
    confusion_matrix.at[predicted[i], actual[i]]=confusion_matrix.at[predicted[i], actual[i]] + 1

# Calculates
accuracy=(cnt*100)/len(predicted)

    
with open('pos-tagging-report.txt', 'w') as f:
# np.savetxt(r'pos-tagging-report.txt', confusion_matrix, fmt='%d',delimiter = "   ")
    f.write("Accuracy: %s" % accuracy+"\n")
    f.write(confusion_matrix.to_string())
    
# print("pos-tagging-report.txt")
# Stores the output in the file called pos-tagging-report.txt
print(confusion_matrix)
print()
print(accuracy)
print("pos-tagging-report.txt")