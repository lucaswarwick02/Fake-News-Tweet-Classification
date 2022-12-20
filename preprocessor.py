from nltk import WordNetLemmatizer, TweetTokenizer
from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()


class Preprocessor:

    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        self.lemmatizer = WordNetLemmatizer()

    def process_features(self, df):
        features = df['tweetText'].progress_apply(lambda text: self._filter_text(text))
        return features

    def process_labels(self, df):
        labels = df['label']
        labels = labels.apply(lambda label: 'fake' if label == 'humor' else label)
        return labels

    def _filter_text(self, text: str) -> str:
        # Before tokenization

        tokens = self.tokenizer.tokenize(text)

        # After tokenization
        tokens = [self._lemmatize_word(word) for word in tokens]

        return ' '.join(tokens)

    def _lemmatize_word(self, word: str) -> str:
        if word[0] == '#':
            return '#' + self.lemmatizer.lemmatize(word[1:])
        else:
            return self.lemmatizer.lemmatize(word)

