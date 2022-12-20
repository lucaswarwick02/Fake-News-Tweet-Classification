from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk import TweetTokenizer
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
logger = logging.getLogger('main')


class Model:

    def __init__(self):
        self.tfidf = TfidfVectorizer(lowercase=False, analyzer='char_wb', ngram_range=(5, 5))
        self.model = MultinomialNB()
        self.tokenizer = TweetTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, df):
        logger.info('Training Model...')

        self.tfidf.fit(df['tweetText'])

        features = self._preprocess_features(df)
        labels = self._preprocess_labels(df)

        self.model.fit(features, labels)
        logger.info('... Done')

    def predict(self, df):
        logger.info('Predicting Labels...')

        features = self._preprocess_features(df)

        y_pred = self.model.predict(features)
        logger.info('... Done')

        return y_pred

    def _preprocess_features(self, df):
        features = df['tweetText'].apply(lambda text: self._filter_text(text))
        features = self.tfidf.transform(features)
        return features

    def _preprocess_labels(self, df):
        labels = df['label']
        labels = labels.apply(lambda label: 'fake' if label == 'humor' else label)
        return labels

    def _filter_text(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text)
        # tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)
