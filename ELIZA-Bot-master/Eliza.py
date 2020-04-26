'''
Eliza by Team GAP
Team Members: Pranav Krishna SJ,Alagappan A, Ganesh Nalluru

Eliza mimics a psychiatrist. It takes input from the user and replies by
spotting words and transforming sentences. The program replies something even 
if the given sentence is not recognizable. It detects repeated inputs, empty
strings and replies accordingly. It has a memory feature which makes the
conversation divert to the previously discussed topic if it does not 
recognize user input. For example, it stores the reply of sentences where the keyword
'my' is used. It detects positive or negative emotions and replies accordingly.
These features are inspired from the Eliza research paper by Joseph Weizanbaum. To
the best of our knowledge, almost all the features of the research paper has been 
implemented in this program.

Reference: weizanbaum, J. (n.d.). ELIZA A Computer Program.


    
PLEASE TYPE 'quit' TO END THE PROGRAM

Algorithm:
    Requests user name and validates.
    Starts conversation.
    Loop:
        Gets user input.
        If user input is quit, the program terminates.
        Checks if input needs to be stored in memory and stores if necessary.
        Matches regular expressions with user inputs.
        If matches:
            Picks a random choice from possible replies to the expression.
            Performs necessary pronoun conversion and prints output.
        If doesn't match:
            Checks memory if anything is stored.
            If something is stored:
                Prints replies in memory or default replies by weighted probability.
                 

Few conversation examples generated through this program are given below.

Eliza: Hi, I'm a psychotherapist. What is your name?
Pranav

Eliza: Hi Pranav, How can I help you today?
Pranav:I am helpless

Eliza:  How long have you been helpless?
Pranav:Since I was born

Eliza:  Please go on Pranav
Pranav:I want help

Eliza:  Do you crave for help?
Pranav:My mother helps me

Eliza:  Does your relationship with your mother related to your feelings today?
Pranav:everytime

Eliza:  That's very interesting Pranav!
Pranav:gibberish

Eliza:  Let's hear more about you mother
Pranav:quit


'''


import re
import random
import numpy as np

#Negative words to be detected
negative = ["sad", "unhappy", "depressed", "sick","dejected","sorrow","regretful"]

#Positive words to be detected
positive = ["happy", "elated", "glad", "better","good","great"]


# Regular expressions for some user inputs whose responses gets stored in memory and used when required
memoryMatchRegEx = {

    r'.* ?(my|our) (?P<keywords>\w+)': ["Lets discuss further about your replacement_text",
                                  "Tell me more about your replacement_text",
                                  "I would like to discuss more about your replacement_text"]
}


#Regular expressions dictionary for spotting words and transforming sentences.
dic = {
       
    #The regular expression below is used for matching uncertain inputs, the responses are general replies a human
    #would give to uncertain responses
    
    r'.* ?(Perhaps|Maybe|I am not sure|I don\'t know).*': ["How certain are you about this?",
                                                           "How sure are you about it?",
                                                           "How often do you respond with uncertainity?",
                                                           "Can't you be more positive"],
       
    #The regular expression below is used for matching words after key words 'I want'. When user input has
    #'I want', it is evident that the user desires something. Sentences like "How would you feel if you 
    #got something", "How badly do you want replacement_text" induces the user to talk more on the topic. 
    #Moreover, these are general questions which fit to any desire
    
    r'I want (?P<keywords>.+)': ["Do you crave replacement_text?",
                                 "Do you really want replacement_text?",
                                 "How would you feel if you got replacement_text",
                                 "How badly do you want replacement_text?"],
       
    #Any reply that has the word 'Sorry' indicates an act of apology. That apology could be to 
    #eliza or about an incdient for which the user feels sorry for etc. Sentences like 
    #"Do you think there was a need for apology?", "How do you feel when somone apologizes you?"
    #would apply to any of these contexts. Therfore they are used as possible replies.
    
    r'.*Sorry ?.*': ["Do you think there was a need for apology?",
                       "How do you feel when somone apologizes you?",
                       "What would you do when you feel sorry?"],
       
    #It is more common for a person to recall about a past experience to a psychatrist. So, the keywords 
    #'I remember' is captured. Sentences transformations like "How often do you think of replacement_text?
    #and "What makes you remember replacement_text?" makes the user believe Eliza is aware of the context
    #that the user is "Thinking" of something.

    r'.*I Remember (?P<keywords>.+)': ["How often do you think of replacement_text?",
                                         "Does thinking of replacement_text bring anything else to mind",
                                         "What makes you remember replacement_text?",
                                         "Why do you want to remember replacement_text now?",
                                         ],
    #The key word do you remember is similar to the one discussed above. Any thing after the 'do you 
    #remember' is captured. General answers to these questions would be what else do you remember. To
    #induce an emotional bonding with the user replies like "Did you think I would forget 
    #something?" are used.  

    r'Do you remember (?P<keywords>.+)': ["Did you think I would forget replacement_text",
                                              "Why do you think I should recall replacement_text now?",
                                              "What else do you remember?"],
    

    #A person coming to a psychotherapist may often talk about the things he envisioned, fantasized or
    #dreamt about. Therfore the keyword dream is captured. Relating that dream with the problem would keep
    #the conversation resticted to user himself. This makes the conversation revolve around the context
    #at which the program is good at. Along with that some general responses 

    r'.* dream (?P<keywords>.+)': ["Do you like to fantasize replacement_text everytime?",
                                    "Do you think \"replacement_text\" has anything to do with your problem?",
                                    "Do you always dream about that?"],
    
    #'I am', 'am I' are frequent replies a user may give, atleast in context of psychotherapy. So
    #those words are captured. The program can use the fact that the user is talking about himself
    #and reply acordingly. For instance, the keyword 'am I' means the user is asking something about
    #himself. By capturing everything the user says after am I, the sentences can be rephrased into
    #one which asks more question about the user based on the captured text.
    
   
    r'.* ?am I (?P<keywords>.+)': ["Do you believe you are replacement_text",
                                  "Would you want to be replacement_text",
                                  "You wish I would tell replacement_text",
                                  "What would it mean if you were replacement_text?"],
    #I am is similar to the prvious one. We know that the user is talking about himself and we can
    #reply acordingly.
   
    r'I am (?P<keywords>.+)':   ["Is it because you are replacement_text you came to me?",
                                  "How long have you been replacement_text?",
                                  "Do you believe it is normal to be replacement_text?"],
    #If the user use 'am' and the context is unknown, some general responses are given.

    r'.* am (?P<keywords>.+)': ["Why do you say 'am'?",
                                 "I don't understand that",
                                 "Can you be more clear"
                                 ],
    #Greetings are common in conversation and it would be appropriate to reply back with greetings.

    r'.* ?(Hello) ?(?P<keywords>.*)?': ["Hey! How's life?",
                                      "Hi! Nice to meet you. Please state your problem.",
                                      "Hello! Let's discuss about your problems."],
   
    #If a user is responding with keyword 'are you', we can derive two facts from it. One is that,
    #user is talking about Eliza and user is asking a question. By leveraging these two facts,we
    #can transform these sentences by using text after you and asking questions acordingly.
    
    r'.* ?are you (?P<keywords>.*)': ["Why are you interested in whether I am replacement_text or not ?",
                                      "Would you prefer if I weren't replacement_text ?",
                                      "Perhaps I am replacement_text in your fantasies",
                                      "Do you sometimes think I am replacement_text"],
    
    #Similar to are you, are they is a question. Replies like 'What if they are something' and 
    #'Possibly they are something' would be highly contextual since it is picking up on where 
    #user left.
    
    r'.* ?are they (?P<keywords>.*)': ["Did you think they might be replacement_text?",
                                       "Would you like if they were not replacement_text?",
                                       "What if they were not replacement_text?",
                                       "Possibly they are replacement_text"],
    
    #'your' could be a common keyword if an user thinks eliza is a human. So text after the 
    #keyword 'your' are captured. Replies like "Why are you concerned over my something?",
    #and "What about your own something?" keeps the conversation personal at which Eliza is
    #good at responding. Sentence transformations like Are you worried about 
    #someone else's replacement_text?" encourages binary responses which Eliza can capture
    #and reply effectively.
    r' .* your (?P<keywords>.*)': ["Why are you concerned over my replacement_text?",
                                    "What about your own replacement_text?",
                                    "Are you worried about someone else's replacement_text?"],
    
    #'I was' is similar to the keyword 'I am'. The user is telling something about himself.
    #"Were you really replacement_text ?" encourages binary output. Moreover, it is general 
    #question which suits to any context. Questins like "Why do you tell 
    #me you were replacement_text now?" are general and suits well to the psychatristic context.
    r'I was (?P<keywords>.*)': ["Were you really replacement_text ?",
                                "Why do you tell me you were replacement_text now?"],
    
    #'Were you' is similar to the keywords 'are you'.
    r'Were you (?P<keywords>.*)': ["Would you like to believe I was replacement_text ?",
                                     "What suggests that I was replacement_text ?",
                                     "Perhaps I was replacement_text",
                                     "What if I had been"],
    r'.*You say (?P<keywords>.+)':["Can you elaborate on replacement_text ",
                              "Do you say replacement_text for some special reason",
                              "That's quite Interesting"
                              ],

    #Since many questions encourage binary responses it is imporant to capture them. Responses
    #are general since there may not be keywords after 'yes' or 'no' to capture.
     r'Yes ?(?P<keywords>.*)':["How often do you reply positively?",
                              "You are sure?",
                              "I see","I Understand"
                              ],
     r'No ?(?P<keywords>.*)':["are you saying 'no' just to be negative?",
                              "you seem bit negative",
                              "why not?","why 'NO'?"
                              ],
    #Presence of the keyword 'Because' indicates there is some kind of 'reason' involved. So the
    #keyword 'reason' is used to manipulate responses.
    r'.*Because (?P<keywords>.+)':["Is that the reason?",
                              "Don't any other reasons come to mind?",
                              "Can you think of any other reasons?",
                              "Does that reason convience you?"
                              ],
    #Again, why don't you is similar to 'are you'. Here, Probably, user is askinng eliza to do or say.
    #So responses are general and similar to the response for any question to eliza. 
    r'Why don\'t you (?P<keywords>.+)':["Do you believe I don't replacement_text ",
                              "Perhaps I will replacement_text in good time",
                              "Should you replacement_text yourself",
                              "You want me to replacement_text"
                              ],
    
    #In this regular expression, keywords related to family are captured. So those key words
    #are used to frame questions about relationship which the user is talking about
    r'.* ?my (?P<keywords>(mother|father|brother|sister|wife)) .*': ["What was your relationship with your replacement_text like?",
                                                                    "How do you feel about your replacement_text ?",
                                                                    "Does your relationship with your replacement_text related to your feelings today?",
                                                                    "Do you have trouble showing affection with your family?"],
    
    #In this regular expression, keywords related to family are captured. So those key words
    #are used to frame questions about relationship which the user is talking about
    r'.* ?(you remind me of|you are) (?P<keywords>.+)': ["What makes you think I am replacement_text?",
                                  "Does it please you to believe I am replacement_text",
                                  "Do you sometimes wish you were replacement_text",
                                  "Perhaps you would like to be replacement_text"],
    
    #'you keyword me' is a special case. Word which comes between you and me are captured. The 
    #word captured is then used to rephrase it as a question. Some general questions like 
    #"Why do you think I replacement_text you?" used the word to paraphrase that word as question.
    r'.* ?(you (?P<keywords>\w+) me)': ["Why do you think I replacement_text you?",
                                  "Do you really belive I replacement_text you?",
                                  "I don't think I replacement_text you"],
    #The below regular expression, captures any question word and gives general outputs.
    r'.* ?(what|why|when|how) .*': ["That's an interesting question",
                                  "How long is this question in your mind?",
                                  "why did you ask that?",
                                  "Why are you asking that question?"], 
    #The below regular expression, captures words which refers to people with generality. We can 
    #ask the user to be more specific.
    r'.* ?(everyone|everybody|none|nobody) .*': ["Can you be more specific?",
                                  "Do you have someone in mind?",
                                  "Are you talking about anyone in particular?"],
    #Similar to previous expression, any replies which have 'always' can be captured and general
    #replies which asks user to be specific can be given.
    r'.* ?Always .*': ["Do you have anything in particular?",
                                  "Is there any exemptions?",
                                  "Do you like to use definitive language?"],

    #The key word like denotes similarity. But it can also denote desire. To solve disambiguity
    #we can include auxilary verbs after which the keyword 'like' in the context of similarity usually
    #follows
    
    r'.* ?(am|is|are|was) like .*': ["How sure are you about the similarity?",
                                  "Do you like to compare things??",
                                  "How did you make the connection?"],

    # Keywords Negative and positive hold customised replies for negative and positive words.for
    # The user input is compared against negative anf positive lists given at the start of the program.
    'negative':["I am sorry to hear you are replacement_text",
                "Do you think coming here will help you not to be replacement_text ?",
                "I'm sure its not pleasant to be replacement_text"],

    'positive':["How have I helped you to be replacement_text ?",
                "Has your treatment made you replacement_text ?",
                "What makes you replacement_text just now ?"]
}

# List of pronouns used for substitution in the later part of the program
pronouns = {"i": "you", "me": "you",
            "you": "me","your": "my",
            "my": "your","was": "were",
            "i'll": "you will", "i've": "you have",
            "myself":"yourself", "yourself": "myself"}


# Name validation
def userNameValidation():
    # Regex expression to find name from a given user input
    nameExp = r'((i\s?am\s?)|(my\s?name\s?is)|(they\s?call\s?me)|(myself))?\s?(?P<fname>\w+)'
    # Getting the user input
    username = input()
    # Removes space before and after a name
    username=username.strip()
    # Checks if the given name is not a special character and removes blank spaces
    if (username.replace(" ", "") and not (set('[~!@#$%^&*()_+{}":;\']+$').intersection(username))):
        # Matching the regex expression against the userinput by ignoring the case
        match = re.match(nameExp, username, re.IGNORECASE)
        # Capturing the first name
        firstName = match.group('fname')
        print("Eliza: Hi " + firstName + ", How can I help you today?")
        # Passing firstname as a parameter to the bot function
        bot(firstName)

    # If userinput violates the constraint a recursive call back to the same function (userNameValidation).
    else:
        print("Eliza: We are not proceeding without your name. Please type your name.")
        userNameValidation()

# Memory function
def memory(userinput):
    # Loop that runs on regular expression from memoryMatchRegEx function
    for regExpressions in memoryMatchRegEx:
        # Matching the regex expression against the userinput by ignoring the case
        memoryMatch = re.match(regExpressions, userinput, re.IGNORECASE)
        # If match found, enters the condition
        if (memoryMatch != None):
            # Captures the key word
            memoryText = memoryMatch.group('keywords')
            # Picks a random reply relating to the regex expression
            memoryReply = random.choice(memoryMatchRegEx[regExpressions])
            # Replacing the replacement_text with the captured key word
            replacedMemoryText = re.sub(r'replacement_text', memoryText, memoryReply)
            # Returns the replaced memory text
            return replacedMemoryText
    # If match not found returns the user input
    return userinput

# Reply function
def eliza_reply(matchText,reference):
    # Picks a random reply from the regex dictionary, relating to the passed reference value
    reply = random.choice(dic[reference])
    # Splits the matchText into separate words in a list( Tokenizing )
    splits = matchText.split()
    # Running a for look on the splits list
    for i in range(0, len(splits)):
        # In case any of the word matches a key in the pronoun dictionary enter the if condition
        if splits[i].lower() in pronouns:
            # Replaces the word by the key value in the pronoun dictionary
            splits[i] = pronouns[splits[i].lower()]
    # Joining the list into a complete string
    splits = " ".join(splits)
    # Returning the replaced text where the replacement_text is substituted with the string "splits"
    return re.sub(r'replacement_text', splits, reply)

# Regex dictionary mapping
def matchdic(userinput):
    # Running a loop through the regex dictionary
    for decompose in dic:
        # Matching the regex expression against the userinput by ignoring the case
        match = re.match(decompose, userinput, re.IGNORECASE)
        # If match found, enters the condition
        if (match != None):
            # Flag is set to 1, if matched
            flag = 1
            # Try block to check if a regex expression in the dictionary consists of a named group(?P<keywords>.+) or not
            try:
                # Enters the try block if named group exist
                match.group("keywords")
                # Captures the key word
                matchText = match.group('keywords')
                matchText = matchText.replace("?","")
                # If the captured word is in the negative list, enters the condition
                if matchText in negative:
                    # Reference is set as negative
                    reference ="negative"
                    # Reply is a call to the eliza_reply function, where the captured word and the reference are sent as parameters.
                    reply = eliza_reply(matchText,reference)
                    # The reply is displayed
                    print("Eliza: ", reply)
                # If the captured word is in the positive list, enters the condition
                elif matchText in positive:
                    # Reference is set as positive
                    reference = "positive"
                    # Reply is a call to the eliza_reply function, where the captured word and the reference are sent as parameters.
                    reply = eliza_reply(matchText,reference)
                    # The reply is displayed
                    print("Eliza: ", reply)
                # If neither positive nor negative enters the else condition
                else:
                    # Reference is set as decompose( for loop iterator)
                    reference = decompose
                    # Reply is a call to the eliza_reply function, where the captured word and the reference are sent as parameters.
                    reply = eliza_reply(matchText, reference)
                    # The reply is displayed
                    print("Eliza: ", reply)

            # If named group does not exist
            except IndexError:
                # Picks a random reply from the regex dictionary, relating to the passed reference value
                reply = random.choice(dic[decompose])
                # The reply is displayed
                print("Eliza: ", reply)
            # Returns flag value
            return flag

    # If match not found flag is set to 0
    flag = 0
    # Returns flag values
    return flag

# Main bot function
def bot(firstName):
    # A dummy variable for the while loop
    x = 1
    # Filler list, consists of replies that need to be used in case a user input is not understood
    filler = ["Tell me more about it " + firstName, "I see", "Please go on " + firstName,
              "That's very interesting " + firstName + "!"]
    # Empty memory list, to append replies relating to memoryMatchRegEx dictionary
    memorydic=[]

    # A tracker is defined to keep a check on repeated user inputs
    inputTracker = None

    # An infinite while loop
    while x != 0:
        # userinput stores the users reply
        userinput = input(firstName.title() + ": ")
        # A flag is set to 0
        flag = 0
        # If user input is empty space enters the condition
        if (userinput == ''):
            print("Eliza: Please say something")
            # Moves back to start of the loop
            continue
        # If user input is quit enters the condition and breaks out of the loop
        if (userinput.lower() == "quit"):
            print("Eliza: I will not say goodbye to you! See you soon!")
            break
        # Checks if user input is same as the inputTracker, if yes enters the loop
        if (userinput == inputTracker):
            # List of responses if the user in repeating his input
            repeatResponse = ["Are you testing me by repeating yourself?", "Please don't repeat yourself.",
                              "What do you expect me to say by repeating yourself?"]
            # Picks a random choice from the above list
            print("Eliza:", random.choice(repeatResponse))
            # Moves back to start of the loop
            continue

        # Memory function is called, with userinput as a parameter. Checks if the userinput matches with regex expression of memoryMatchRegEx
        text=memory(userinput)

        # If the returned text is not same as the userinput, then text is appended to the memorydic list
        if(text!=userinput):
            memorydic.append(text)

        # Each iteration stores the userinput in inputTracker
        inputTracker = userinput

        # Matchdic function returns flag value. If regex exists in the dictionary, flag is set to 1 else flag by default is 0
        flag = matchdic(userinput)

        # If flag is 0 and memordic list is empty, a random reply is generated from the filler
        if (flag == 0 and len(memorydic)==0):
            reply = np.random.choice(filler,replace=False)
            print("Eliza: ", reply)

        # If memorydic is not empty and flag is 0, then filler list and memorydic is combined as a single list.
        # A probability of 0.6 is set for memorydic list and 0.4 for filler list, from which reply is picked and displayed.
        elif (len(memorydic)>0 and flag == 0):
            combinedList=[memorydic,filler]
            listChoice = np.random.choice(len(combinedList),replace=False,p=[0.6,0.4])
            reply=np.random.choice(combinedList[listChoice],replace=False)
            print("Eliza: ", reply)

# Main function
if (__name__ == "__main__"):
    print("Eliza: Hi, I'm a psychotherapist. What is your name?")
    # Initial call to the name validation function
    userNameValidation()
