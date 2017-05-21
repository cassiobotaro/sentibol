from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np


def compare_tweet_jaccard(tweet, tweet_compare):
    words_tweet = tweet.lower().split()
    words_compare = tweet_compare.lower().split()
    intersection = set(words_tweet).intersection(set(words_compare))
    intersection_size = len(intersection)
    calc = float(intersection_size) / (len(words_tweet) + len(words_compare)
                                       - intersection_size)

    if calc >= 0.20:  # Define o quao semelhante é um tweet com o outro
        return True  # se tweet igual a tweet_compare
    return False


tweets = ['o cruzeiro jogou muito bem', 'parabéns pela vitória cruzeiro',
          'ainda acho o cruzeiro ruim', 'o cruzeiro não jogou bem',
          'cruzeiro jogou contra o sport',
          'primeira rodada o cruzeiro enfrentou o sport',
          'bela vitória do cruzeiro', 'mais uma vitória do cruzeiro']

tf_vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('portuguese'), analyzer='word',
            ngram_range=(1, 1), lowercase=True, use_idf=True)
matrix = tf_vectorizer.fit_transform(tweets)

feature_array = np.array(tf_vectorizer.get_feature_names())
tfidf_sorting = np.argsort(matrix.toarray()).flatten()[::-1]
feature_words = feature_array[tfidf_sorting][:10]
featured_tweets = []
for tweet in tweets:
    relevance = 0
    for word in feature_words:
        if word in tweet:
            relevance += 1
    if relevance > 3:
        featured_tweets.append((tweet, relevance))

    print(f'{tweet} -> {relevance}')

# eliminate duplicated tweets
non_duplicated = []
for t in sorted(featured_tweets, key=itemgetter(1), reverse=True):
    if not non_duplicated:
        non_duplicated.append(t[0])
    else:
        for nd in non_duplicated:
            if not compare_tweet_jaccard(nd, t[0]):
                non_duplicated.append(t[0])


print(80 * '-')
print('Tweets mais relevantes:')
print(non_duplicated)
