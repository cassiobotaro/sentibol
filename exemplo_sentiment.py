# lists based on https://en.wikipedia.org/wiki/List_of_emoticons
# Versão original removia pontuação(o que é errado,
# pois emoticons são feitos utilizando pontuação)
# falta de emojis
# deveria trazer as palavras a sua raiz
from emoticons import (positive_emoticons, negative_emoticons,
                       positive_sentiment, negative_sentiment)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

list_emoticons_positive = positive_emoticons.split('\n')
list_emoticons_negative = negative_emoticons.split('\n')
list_negative = negative_sentiment.split('\n')
list_positive = positive_sentiment.split('\n')


def classify(text):
    for t in text.split(' '):
        if t in list_emoticons_positive:
            return 1
        elif t in list_emoticons_negative:
            return -1
    return votation(text)


def votation(text):
    vote = 0
    text = clean_text(text)

    for t in text:
        if t in list_negative:
            vote = vote - 1
        elif t in list_positive:
            vote = vote + 1

    if vote > 1:
        return 1
    elif vote < 0:
        return -1
    else:
        return vote


def clean_text(text):
    token_text = word_tokenize(text.lower())
    text = (word for word in token_text
            if word not in stopwords.words('portuguese'))
    return text


def predict(tweets):
    return np.array([classify(phrase) for phrase in tweets])


tweets = ['cruzeiro tem um bom time', 'acompanhe os jogos da primeira rodada',
          'muito ruim esse jogo do cruzeiro', 'cruzeiro! :)',
          'Jogo morno entre cruzeiro e são paulo',
          'Meu são paulo perdeu hoje :\'(']

predictions = predict(tweets)
for prediction, tweet in zip(predictions, tweets):
    print(f'{tweet} -> {prediction}')
