import json
import ast
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections, re
import urllib
import string
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models

ps = PorterStemmer()
stops = set(stopwords.words("english"))

def review_word(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stops]
    stemmed_words=[ps.stem(w) for w in meaningful_words]
    return stemmed_words

dict = {}
i = 0
with open('data.txt') as data:
    for line in data:
        line = ast.literal_eval(line)
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
             if (pos == 'NN')]#or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    wd.append(review_word(' '.join(nouns)))

#PROGRESS : List of list of nouns in a review

dictionary = corpora.Dictionary(wd)
corpus = [dictionary.doc2bow(text) for text in wd]
ldamodel = models.LdaModel(corpus, id2word=dictionary,num_topics=3)
print(ldamodel.print_topics(num_topics=3, num_words=4))

#PROGRESS : LDA Successfully implemented :)

