from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from model import Model
import pandas as pd
import datetime
import logging
import os

# Define the location constants
FOLDER_NAME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUTPUT_PATH = os.path.join('./out', FOLDER_NAME)
# Create the output folder
if not os.path.exists('./out'):
    os.mkdir('./out')
os.mkdir(OUTPUT_PATH)
# Set up the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S')
logger = logging.getLogger('main')
# Set up the logger's file handler
if not os.path.exists(os.path.join(OUTPUT_PATH, 'output.log')):
    open(os.path.join(OUTPUT_PATH, 'output.log'), 'w').close()
fh = logging.FileHandler(os.path.join(OUTPUT_PATH, 'output.log'))
logger.addHandler(fh)


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
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f'Accuracy Score = {accuracy:.4f}')
    logger.info(f'F1 Score = {f1:.4f}')

    # Confusion Matrix
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    cm.ax_.set_title(f'f1 score = {f1:.4f}')
    cm.figure_.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))
    logger.info('Confusion Matrix saved to confusion_matrix.png')

    # Classification Report
    with open(os.path.join(OUTPUT_PATH, 'classification_report.txt'), 'w') as f:
        print(classification_report(y_test, y_pred, zero_division=0), file=f)
    logger.info('Classification Report saved to classification_report.txt')


def custom_accuracy(y_test, y_pred):
    positive_label = 'fake'
    negative_label = 'real'

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_test[i] == positive_label and y_pred[i] == positive_label:
            TP += 1
        if y_test[i] == negative_label and y_pred[i] == positive_label:
            FP += 1
        if y_test[i] == negative_label and y_pred[i] == negative_label:
            TN += 1
        if y_test[i] == positive_label and y_pred[i] == negative_label:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    logger.info(f'accuracy = {accuracy:.4f}')

    logger.info('fake:')
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    logger.info(f'... precision = {precision:.4f}')
    logger.info(f'... recall = {recall:.4f}')
    logger.info(f'... f1 = {f1:.4f}')

    logger.info('real:')
    precision = TN / (TN + FN)
    recall = TN / (TN + FP)
    f1 = 2 * ((precision * recall) / (precision + recall))
    logger.info(f'... precision = {precision:.4f}')
    logger.info(f'... recall = {recall:.4f}')
    logger.info(f'... f1 = {f1:.4f}')


if __name__ == '__main__':
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

    custom_accuracy(y_test, y_pred)

