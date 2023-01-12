import pandas as pd
from nltk import TweetTokenizer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
import re


class Model:

    def __init__(self):
        self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(5, 5))
        self.model = MultinomialNB()
        self.tokenizer = TweetTokenizer(preserve_case=False)
        self.lemmatizer = WordNetLemmatizer()

        self.url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        self.emoji_regex = re.compile("["
                                      u"\U0001F600-\U0001F64F"  # emoticons
                                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                      u"\U00002702-\U000027B0"
                                      u"\U000024C2-\U0001F251"
                                      "]+", flags=re.UNICODE)

        self.emojis_scalar = MinMaxScaler()
        self.hashtags_scalar = MinMaxScaler()

    def fit(self, df: pd.DataFrame):
        """
        Fit the model from a given training set.

        Args:
            df: Training dataset
        """
        df = self._extract_features(df)

        df.drop_duplicates(subset=['filteredTweetText'], keep='first', inplace=True, ignore_index=False)
        df['label'] = df['label'].apply(lambda label: 'fake' if label == 'humor' else label)

        features = self._generate_features(df, fit_tfidf=True)

        # Train the Model
        self.model.fit(features, df['label'])

    def predict(self, df: pd.DataFrame) -> list[str]:
        """
        Predict on the features from the testing set.

        Args:
            df: Testing dataset

        Returns:
            List of predicted labels
        """
        df = self._extract_features(df)

        features = self._generate_features(df, fit_tfidf=False)

        # Predict on the Features
        return self.model.predict(features)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['emojis'] = df['tweetText'].apply(lambda text: self._count_emojis(text))
        df['hashtags'] = df['tweetText'].apply(lambda text: self._count_hashtags(text))
        df['filteredTweetText'] = df['tweetText'].apply(lambda text: self._filter_text(text))

        return df

    def _generate_features(self, df: pd.DataFrame, fit_tfidf: bool):
        if fit_tfidf:
            return hstack([
                self.tfidf.fit_transform(df['filteredTweetText']),
                self.emojis_scalar.fit_transform(df['emojis'].values.reshape(-1, 1)),
                self.hashtags_scalar.fit_transform(df['hashtags'].values.reshape(-1, 1))
            ])
        else:
            return hstack([
                self.tfidf.transform(df['filteredTweetText']),
                self.emojis_scalar.transform(df['emojis'].values.reshape(-1, 1)),
                self.hashtags_scalar.transform(df['hashtags'].values.reshape(-1, 1))
            ])

    def _filter_text(self, text: str) -> str:
        text = re.sub(self.emoji_regex, '', text)
        tokens = self.tokenizer.tokenize(text)
        tokens = [self._lemmatize_word(word) for word in tokens]

        return ' '.join(tokens)

    def _lemmatize_word(self, word: str) -> str:
        if word[0] == '#':
            return '#' + self.lemmatizer.lemmatize(word[1:])
        else:
            return self.lemmatizer.lemmatize(word)

    def _count_emojis(self, text: str) -> int:
        return len(re.findall(self.emoji_regex, text))

    def _count_hashtags(self, text: str) -> int:
        return len([word for word in self.tokenizer.tokenize(text) if word[0] == '#'])
