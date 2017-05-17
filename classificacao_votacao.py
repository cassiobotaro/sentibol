from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

classificator = VotingClassifier(estimators=[
    ('svm', SVC(kernel='linear')),
    ('naive', MultinomialNB(alpha=.01))]
)

training_set = ['esse cruzeiro jogou muito', 'parabéns pela vitória cruzeiro',
                'ainda acho o cruzeiro ruim', 'o cruzeiro não jogou bem',
                'cruzeiro jogou contra o são paulo',
                'primeira rodada o cruzeiro enfrentou o são paulo']
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
