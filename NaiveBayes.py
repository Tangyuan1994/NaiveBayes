import numpy as np 
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.tokenize import word_tokenize
import collections
data=pd.read_csv("Tweets.csv")
data= data.copy()[['airline_sentiment', 'text']]
# remove the punction and stopwords
def review_to_words( review ):
    review_text = review
    no_punctions = re.sub("[^a-zA-Z]", " ", review_text) 
    wordslower= no_punctions.lower()
    words = word_tokenize(wordslower)  
    stopswd = set(stopwords.words("english"))                  
    meaningful_wd = [w for w in words if not w in stopswd]
    return(meaningful_wd)

clean_text = []
for tweet in data['text']:
    clean= review_to_words(tweet)
    clean_text.append(clean)

data['text'] = clean_text
# 80% of data is data_train for traing,20% of data is data_test for testing
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['airline_sentiment'], test_size=0.20, random_state=0)
data_train = pd.DataFrame()
data_test = pd.DataFrame()
data_train['text'] = X_train
data_train['airline_sentiment'] = y_train
data_train = data_train.reset_index(drop=True)
data_test['text'] = X_test
data_test['airline_sentiment'] = y_test
data_test = data_test.reset_index(drop=True)

# d is the number of words in positive text,I remember d= 16650
# e is the number of words in negative text,I remember e= 80297
# f is is the number of words in neutral text,I remember f= 21642
a=0
b=0
c=0
e=0
d=0
f=0
dataAll=data_train['text'] 
copos1=collections.Counter()
coneg2=collections.Counter()
coneu3=collections.Counter() 
dataPos=data_train.copy()[data_train.airline_sentiment == 'positive']
dataNeg=data_train.copy()[data_train.airline_sentiment == 'negative']
dataNeu=data_train.copy()[data_train.airline_sentiment == 'neutral']
pos_words =(dataPos['text'].tolist())
neg_words =(dataNeg['text'].tolist())
neu_words =(dataNeu['text'].tolist())
all_words =(data_train['text'].tolist())

# Attention de bien differencier les nombres de tweets des nombres de mots
print("nuttpos",len(pos_words),sum([len(u) for u in pos_words]))
print("nuttneg",len(neg_words),sum([len(u) for u in pos_words]))
print("nuttneu",len(neu_words),sum([len(u) for u in pos_words]))

#calcul the probability of positive text,negative test,and neutral text
#I remeber P_pos=0.1638,P_neg=0.623975409836,P_neu=0.212175546448

P_pos = float(len(dataPos))/len(dataAll)
P_neg = float(len(dataNeg))/len(dataAll)
P_neu = float(len(dataNeu))/len(dataAll)
print("priors",P_pos,P_neg,P_neu)

# I use copos,coneg,coneu to divide the training data into 3 emotions
for i in range(0, len(pos_words)):
        copos1.update(pos_words[i])

for i in range(0, len(neg_words)):
        coneg2.update(neg_words[i])

for i in range(0, len(neu_words)): 
        coneu3.update(neu_words[i])

voctot = collections.Counter()
voctot.update(copos1)
voctot.update(coneg2)
voctot.update(coneu3)
nvoctot=len(voctot)
print("nvoctot",nvoctot)
for w in voctot.keys():
    print("VOC "+w)

#calcul the number d,e,f
for i in range(0, len(pos_words)):
       for w in pos_words[i]:
                d=d+1

for i in range(0, len(neg_words)):
       for w in neg_words[i]:
                e=e+1

for i in range(0, len(neu_words)):
       for w in neu_words[i]:
                f=f+1
print("nmots pos neg neu",d,e,f)

#then I predict the result of Testing data
# the result of predicting including positive,negative and neutral
class_choice = ['positive', 'negative', 'neutral']
classification = []
test_words =(data_test['text'].tolist())
# i= the i-th person'sopinion
for i in range(0, len(test_words)):
#each word in i-th person's opinion
       for w in test_words[i]:
#I think this part maybe have some problems, I use the formulas in the Wikipedia, but there are lots of variables
#So i'm afraid of this part has some problems,maybe we can talk about this part on monday.

# vous voulez calculer P(w|S)=P(w,S)/P(S)=(nb d'occ de w pos / nb de mots tot)/(nb de mots pos / nb de mots tot) = (nb d'occ de w pos/nb de mots pos)
# copos1[w] = nb d'occ de w pos
# d = nb de mots pos
# WARNING: pourquoi +1 ?
# WARNING: n'oubliez pas le float() devant copos1
# la, vous calculez: sum_w ln(P(w|S))
# si vous utilisez du smoothing, il vaut mieux calculer de vrais probas ici, c'est plus clair que d'essayer de deporter le denominateur a l'exterieur
               a=a+np.log((float(copos1[w]+1))/float(d+nvoctot))
               b=b+np.log((float(coneg2[w]+1))/float(e+nvoctot))
               c=c+np.log((float(coneu3[w]+1))/float(f+nvoctot))
# il suffit de calculer ln P(S|D) = sum_w ln(P(w|S)) + ln(P(S)) - constante
# et on se moque de la constante
       a=a+np.log(P_pos)
       b=b+np.log(P_neg)
       c=c+np.log(P_neu)
#I choose the best results from the training data to predict the testing data
       probability = (a, b, c)
       classification.append(class_choice[np.argmax(probability)])
       a=0
       b=0
       c=0
#I calcul the accuracy
compare = []
for i in range(0,len(classification)):
            if classification[i] == data_test.airline_sentiment.tolist()[i]:
                value ='correct'
                compare.append(value)
            else:
                value ='incorrect'
                compare.append(value)

r = Counter(compare)
accuracy = float(r['correct'])/(r['correct']+r['incorrect'])
print accuracy


