from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier


class Classifier:
    def __init__(self, model, scale_data=True, dummy_encoding=True):
        self.model = model
        self.dummy_encoding = dummy_encoding
        self.scale_data = scale_data
        self.encoder = None
        self.scaler = None


def make_classifiers():
    """Creates a list of classifiers to train."""

    return [
        # random forest
        Classifier(RandomForestClassifier(), dummy_encoding=True, scale_data=True),
        Classifier(RandomForestClassifier(), dummy_encoding=True, scale_data=False),
        Classifier(RandomForestClassifier(), dummy_encoding=False, scale_data=True),
        Classifier(RandomForestClassifier(), dummy_encoding=False, scale_data=False),

        # stochastic gradient descent
        Classifier(SGDClassifier(loss="log"), dummy_encoding=True, scale_data=True),
        Classifier(SGDClassifier(loss="log"), dummy_encoding=True, scale_data=False),
        Classifier(SGDClassifier(loss="log"), dummy_encoding=False, scale_data=True),
        Classifier(SGDClassifier(loss="log"), dummy_encoding=False, scale_data=False),

        # max entropy
        Classifier(LogisticRegression(), dummy_encoding=True, scale_data=True),
        Classifier(LogisticRegression(), dummy_encoding=True, scale_data=False),
        Classifier(LogisticRegression(), dummy_encoding=False, scale_data=True),
        Classifier(LogisticRegression(), dummy_encoding=False, scale_data=False),

        # ANN
        Classifier(MLPClassifier(), dummy_encoding=True),
        Classifier(MLPClassifier(), dummy_encoding=True, scale_data=False),
    ]
