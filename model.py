from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from preprocessor import Preprocessor


class Model:

    def __init__(self):
        self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(5, 5))
        self.model = MultinomialNB()
        self.preprocessor = Preprocessor()

    def fit(self, df):
        # Training Preprocessing
        df.drop_duplicates(subset=['tweetText'], keep='first', inplace=True, ignore_index = False)

        features = self.preprocessor.process_features(df)
        labels = self.preprocessor.process_labels(df)

        tfidf_features = self.tfidf.fit_transform(features)

        self.model.fit(tfidf_features, labels)

    def predict(self, df):
        features = self.preprocessor.process_features(df)

        tfidf_features = self.tfidf.transform(features)

        return self.model.predict(tfidf_features)
