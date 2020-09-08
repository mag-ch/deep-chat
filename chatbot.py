
#Meet Robo: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# uncomment the following only the first time
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only


#Reading in the corpus
with open('fanfics.txt', 'r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

raw = re.findall(r'"(.*?)"', raw)
raw = " ".join(raw)
raw = raw.replace(",",".")

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words


# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx + 1]
        robo_response = robo_response.strip("\"")
        return robo_response

def get_robot_response(user_response):
    # user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if("thank" in user_response ):
            # print("ROBO: You are welcome..")
            return "You're welcome :)"
        else:
            if(greeting(user_response)!=None):
                # print("ROBO: "+greeting(user_response))
                return greeting(user_response)
            else:
                # print("ROBO: ",end="")
                res = response(user_response)
                # print(response(user_response))
                sent_tokens.remove(user_response)
                return res
    else:
        # print("ROBO: Bye! take care..")
        return "see ya bish"

def format_pronouns(robo_response, name):
    tokens = nltk.tokenize.word_tokenize(robo_response)
    pos = nltk.pos_tag(tokens)
    # pos = nltk.pos_tag(robo_response.split())
    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1:  # avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []

    print (person_list)

def format_response(response):
    response_dict = {
        1: ['she', 'he']
    }

text = """
Some economists have responded positively to Bitcoin, including 
Francois R. Velde, senior economist of the Federal Reserve in Chicago 
who described it as "an elegant solution to the problem of creating a 
digital currency." In November 2013 Richard Branson announced that 
Virgin Galactic would accept Bitcoin as payment, saying that he had invested 
in Bitcoin and found it "fascinating how a whole new global currency 
has been created", encouraging others to also invest in Bitcoin.
Other economists commenting on Bitcoin have been critical. 
Economist Paul Krugman has suggested that the structure of the currency 
incentivizes hoarding and that its value derives from the expectation that 
others will accept it as payment. Economist Larry Summers has expressed 
a "wait and see" attitude when it comes to Bitcoin. Nick Colas, a market 
strategist for ConvergEx Group, has remarked on the effect of increasing 
use of Bitcoin and its restricted supply, noting, "When incremental 
adoption meets relatively fixed supply, it should be no surprise that 
prices go up. And that’s exactly what is happening to BTC prices."
"""

format_pronouns(text, "mag")
