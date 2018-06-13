from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from time import time
import numpy as np

import classifier.file_handler as fh
from classifier.classifier_factory import make_classifiers
from classifier.data_loader import get_normalised_data
from classifier.accuracy_score import mean_absolute_error_score


def pick_best_model(classifier_list, X, y, folds):
    def cross_validate_model(X, y):
        start = time()

        if classifier.scale_data:
            clf = make_pipeline(StandardScaler(), classifier.model)
        else:
            clf = classifier.model

        if not classifier.dummy_encoding:
            location_column = X[:, [2]]

            one_hot_encoded_location = OneHotEncoder().fit_transform(location_column).toarray()

            X = np.hstack((X[:, :2], one_hot_encoded_location))

        scores = cross_val_score(clf, X, y, cv=folds, scoring=mean_absolute_error_score)
        accuracy_score = scores.mean()

        print('%d-fold CV time: %.3f seconds' % (folds, time() - start))
        print('accuracy: %.3f%%' % (accuracy_score * 100))

        return accuracy_score

    best_accuracy = 0
    best_model = None

    for classifier in classifier_list:
        print('*' * 50)
        print('classifier:', classifier.model.__class__.__name__)
        print('scale_data: %s, dummy_encoding: %s' % (classifier.scale_data, classifier.dummy_encoding))

        accuracy_score = cross_validate_model(X, y)

        if accuracy_score > best_accuracy:
            best_accuracy, best_model = accuracy_score, classifier

    print('*' * 50)
    print('best model:', best_model.model.__class__.__name__)
    print('accuracy: %.3f%%' % (best_accuracy * 100))

    return best_model


X, y = get_normalised_data()

classifier = pick_best_model(make_classifiers(), X, y, folds=10)

if classifier.scale_data:
    classifier.scaler = StandardScaler()
    classifier.scaler.fit(X)
    X = classifier.scaler.transform(X)

if not classifier.dummy_encoding:
    location_column = X[:, [2]]

    classifier.encoder = OneHotEncoder()
    one_hot_encoded_location = classifier.encoder.fit(location_column)

    X = np.hstack((X[:, :2], one_hot_encoded_location))

classifier.model.fit(X, y)

# only save the "best" model to the disk
fh.save_model(classifier, fh.crime_classifier_path())
