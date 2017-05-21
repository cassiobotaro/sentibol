from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


classificator = SVC(kernel='linear')
training_set = ['o cruzeiro jogou muito bem', 'parabéns pela vitória cruzeiro',
                'ainda acho o cruzeiro ruim', 'o cruzeiro não jogou bem',
                'cruzeiro jogou contra o sport',
                'primeira rodada o cruzeiro enfrentou o sport']
labels = [1, 1, -1, -1, 0, 0]
tf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'),
                                analyzer='word', ngram_range=(1, 1),
                                lowercase=True, use_idf=True,
                                strip_accents='unicode')

features = tf_vectorizer.fit_transform(training_set)
classificator.fit(features, labels)
tweets = ['cruzeiro tem um bom time', 'acompanhe os jogos da primeira rodada',
          'muito ruim esse jogo do cruzeiro']
vector_tweets = tf_vectorizer.transform(tweets)
predictions = classificator.predict(vector_tweets)
for prediction, tweet in zip(predictions, tweets):
    print(f'{tweet} -> {prediction}')
