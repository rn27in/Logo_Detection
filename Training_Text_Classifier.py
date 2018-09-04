from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import numpy as np


def generate_sentences(comments):
    ''' 
    :param comments: Takes a set of comments and generates multiples comments using synonymns from wordnet
    :return: a tuple of generated sentence and a dictionary of synonymns#########
    ''''''
    '''
    num_words = [word_tokenize(elem) for elem in comments]
    num_words = [elem.lower() for outer in num_words for elem in outer]
    num_words = [lemmatizer.lemmatize(elem) for elem in num_words]
    #The synsets will not look for stop words###########
    num_words = [elem for elem in num_words if elem not in stop]
    num_words = set(num_words)
    num_dict = {}
    for elem in num_words:
        for ss in wordnet.synsets(elem):
            temp = ss.lemma_names()
            num_dict[elem] = temp

    final_tokens = []
    for sent in comments:
        temp = word_tokenize(sent)
        temp_list = []
        for elem in temp:
            elem = re.sub(r"'s", '', elem)
            elem = elem.lower()
            if elem in stop:
                temp_list.append([elem.lower()])
            else:
                try:
                    elem = lemmatizer.lemmatize(elem)
                    word_list = num_dict[elem]
                    temp_list.append(word_list)
                except KeyError:
                    temp_list.append([elem])

        final_tokens.append(temp_list)

    list_sentences = []
    for sent in final_tokens:
        temp = list(itertools.product(*sent))
        list_sentences.append(temp)

    return ([' '.join(elem) for inter in list_sentences for elem in inter], num_dict)

def text_classifier(text, comments_counts, comments_bin):
    '''
    
    :param text: text on which classification required 
    :param comments_counts: positive class records
    :param comments_bin: negative class records
    :return: Writes the vectorizer and classifier to disk
    '''
    vectorizer = TfidfVectorizer()
    input_matrix = vectorizer.fit_transform(text)
    labels_text = np.array(list(np.repeat(1, len(comments_counts))) + list(np.repeat(0, len(comments_bin))))
    joblib.dump(vectorizer, '../vect_text')
    clf = SGDClassifier(penalty='l2', loss='log')
    clf.fit(input_matrix, labels_text)

    joblib.dump(clf, '../clf_text')


if __name__ == '__main__':
    comments_num = [u'How many Targetx logos are there ?', u'# of logos of Targetx ?',u'Num of logos of Targetx ?', u'What is the no. Of logos of Targetx in the image ?',
    u'Count of logogram of Targetx in the image ?']

    comments_binary = [u'Is there a Targetx logo in the photo ?',
    u'Do we have any logo of Targetx ?',
     u'Does this photo have any logo of Targetx ?',
    u'Do you think this photo has any logo of Targetx ?']

    lemmatizer = WordNetLemmatizer()
    comments_counts,counts_dict = generate_sentences(comments_num)
    comments_bin, bin_dict = generate_sentences(comments_binary)
    final_comments = comments_counts + comments_bin

    text_classifier(final_comments,comments_num,comments_binary)




