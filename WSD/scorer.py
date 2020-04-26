
# Importing the required packages
import numpy as np
import pandas as pd
import sys

# User command line arguments are stored in variables
# Stored values are open and read
my_list=sys.argv[1]
key=sys.argv[2]

my_list=open(my_list)
my_list=my_list.read()

key=open(key)
key=key.read()

# Key and predicted senses are split into list
key=key.split()
my_list=my_list.split()

# only the senses are extracted from the list and appended in a list
ans_key=[]
my_ans=[]
for i in range(0,len(key)):
    if(key[i].find("senseid")==False):
        ans_key.append(key[i])
        my_ans.append(my_list[i])


# Confusion matrix
initialList=np.zeros((2,2))
confusionMatrix=pd.DataFrame(initialList, index=["phone","product"], columns=["phone","product"])
acc=0
# Comparison between the key and the predicted list
# the values are stored in a confusion matrix as a data frame
# the accuracy is calculated based the confusion matrix
for i in range(0,len(ans_key)):
    if(ans_key[i]==my_ans[i]):
        acc+=1
        actual=(ans_key[i].split("="))[1].replace('"','')[:-2]
        predicted=(my_ans[i].split("="))[1].replace('"','')[:-2]
        count=confusionMatrix.at[actual,predicted]
        count+=1
        confusionMatrix.at[actual,predicted]=count
    else:
        actual=(ans_key[i].split("="))[1].replace('"','')[:-2]
        predicted=(my_ans[i].split("="))[1].replace('"','')[:-2]
        count=confusionMatrix.at[actual,predicted]
        count+=1
        confusionMatrix.at[actual,predicted]=count

# results
print()
print('Confusion Matrix: ')
print(confusionMatrix)
print()
print('Accuracy: ',acc/len(ans_key))
