from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from model import Model
import pandas as pd
import datetime
import logging
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
logger = logging.getLogger('main')


def get_training_data() -> pd.DataFrame:
    logger.info('Loading Training Data...')
    df = pd.read_csv('./dataset/mediaeval-2015-trainingset.txt', sep='\t', index_col=None)
    logger.info('... Done')
    return df


def get_testing_data() -> pd.DataFrame:
    logger.info('Loading Testing Data...')
    df = pd.read_csv('./dataset/mediaeval-2015-testset.txt', sep='\t', index_col=None)
    logger.info('... Done')
    return df


def evaluate(y_test, y_pred):
    # Accuracy Score
    accuracy = round(accuracy_score(y_test, y_pred), 3)

    # Confusion Matrix
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    cm.ax_.set_title(f'Accuracy = {accuracy}')
    cm.figure_.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    logger.info('Confusion Matrix saved to confusion_matrix.png')

    # Classification Report
    with open(os.path.join(output_path, 'classification_report.txt'), 'w') as f:
        print(classification_report(y_test, y_pred, zero_division=0), file=f)
    logger.info('Classification Report saved to classification_report.txt')


if __name__ == '__main__':
    # Create Output Folder
    folder_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join('./out', folder_name)
    if not os.path.exists('./out'):
        os.mkdir('./out')
    os.mkdir(output_path)

    # Load Training and Testing Sets
    train_df = get_training_data()
    test_df = get_testing_data()

    # Train the Model
    model = Model()
    model.fit(train_df)

    # Predict on the test set
    y_pred = model.predict(test_df)
    y_test = test_df['label']

    evaluate(y_test, y_pred)

