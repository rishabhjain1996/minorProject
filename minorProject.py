import json
import ast
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections, re
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
import pandas as pd
import numpy as np

ps = PorterStemmer()
stops = set(stopwords.words("english"))


def review_word(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stops]
    stemmed_words = [ps.stem(w) for w in meaningful_words]
    return stemmed_words


dict = {}
i = 0
with open('data.txt') as data:
    for line in data:
        # print line
        line = ast.literal_eval(line)
        #  print line
        #  exit(1)
        json_data = json.loads(json.dumps(line))
        dict[i] = json_data
        i += 1

l = []
wd = []
for i in dict:
    for j in dict[i]:
        if j == 'reviewText':
            l.append(str(dict[i][j]))

for i in l:
    text = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(text)
    nouns = [word for word, pos in tagged
             if (pos == 'NN')]  # or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    wd.append(review_word(' '.join(nouns)))

# PROGRESS : List of list of nouns in a review


dictionary = corpora.Dictionary(wd)
corpus = [dictionary.doc2bow(text) for text in wd]

features = []

if __name__ == "__main__":
    ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=10, passes=200)
    features.append(ldamodel.show_topics(num_topics=5, num_words=4, log=False, formatted=False))

# PROGRESS : LDA Successfully implemented :)


# read ANEW csv file with Pandas and create dictionary
dict_valence = {}
dict_arousal = {}
dict_dominance = {}
df = pd.read_csv('all.csv')
for i in np.arange(len(df)):
    dict_valence[df.loc[i, 'Description']] = df.loc[i, 'Valence Mean'] / 5 - 1
    dict_arousal[df.loc[i, 'Description']] = df.loc[i, 'Arousal Mean'] / 5 - 1
    dict_dominance[df.loc[i, 'Description']] = df.loc[i, 'Dominance Mean'] / 5 - 1

l2 = []
for i in l:
    l2.append(review_word(i))


pos_valence = []
neg_valence = []
pos_arousal = []
neg_arousal = []
pos_dominance = []
neg_dominance = []

for i in l2:
    v_pos = v_neg = a_pos = a_neg = d_pos = d_neg = 0

    for j in i:

        if j not in dict_valence.keys():
            continue;

        val = dict_valence[j]
        if val >= 0:
            v_pos += val
        else:
            v_neg -= val

        val = dict_dominance[j]

        if val >= 0:
            d_pos += val
        else:
            d_neg -= val

        val = dict_arousal[j]
        if val >= 0:
            a_pos += val
        else:
            a_neg -= val

    pos_valence.append(v_pos)
    neg_valence.append(v_neg)
    pos_arousal.append(a_pos)
    neg_arousal.append(a_neg)
    pos_dominance.append(d_pos)
    neg_dominance.append(d_neg)

s = set()
for i in features[0]:
    for j in i[1]:
        s.add(j[0])




datas = []
for i in range(len(l2)):
    l4 = []
    for j in s:
        l3 = []
        temp = l2[i].count(j) / len(l2[i])
        l3.append(temp * pos_valence[i])
        l3.append(temp * pos_arousal[i])
        l3.append(temp * pos_dominance[i])
        l3.append(temp * neg_valence[i])
        l3.append(temp * neg_arousal[i])
        l3.append(temp * neg_dominance[i])
        l4.append(l3)
    datas.append(l4)


# dimensions are no of reviews x no. of features for every sentiment
# dictionary [ review(100000) ] [feature(10) ] [ sentiment (6)]

k_set = []
maxvp = []
maxvn = []
for i in range(len(s)):
    temp = []
    for j in range(6):
        temp.append(0)
    maxvp.append(temp)
    maxvn.append(temp)


# dimensions are no. of features x six (no of sentiments)

def update(rev_id):
    global maxvp
    global maxvn

    k_set.append(rev_id)  # add element in the set

    # update it
    for i in range(len(datas[rev_id])):
        for j in range(3):
            maxvp[i][j] = max(maxvp[i][j], datas[rev_id][i][j])
        for j in range(3, 6):
            maxvn[i][j] = max(maxvn[i][j], datas[rev_id][i][j])

kval = 5
#no of top reviews that we want to display

while len(k_set) < kval:
    d = 0
    ind = 0
    for i in range(len(datas)):
        dtemp = 0
        for j in range(len(datas[i])):
            for k in range(3):
                dtemp += max(0, (datas[i][j][k] - (maxvp[j][k]))) +  max(0, (datas[i][j][k+3]  - maxvn[j][k] ))

        if dtemp > d:
            d = dtemp
            ind = i

    update(ind)


for i in k_set:
    print(l[i])
    print()
