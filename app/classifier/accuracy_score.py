import numpy as np


def mean_absolute_error_score(estimator, X_test, y_test):
    """
    Calculates a mean absolute error based on the predict_proba prediction values.

    :param estimator: Trained scikit-learn model instance. Must have a predict_proba method.
    :param X_test: Array-like object of training data
    :param y_test: Array-like object of labels for the training data
    :return: MAE score (between 0 and 1) to maximise
    """
    observations = _samples_to_proba(X_test, y_test)

    predictions = estimator.predict_proba(X_test)

    return 1 - np.mean(np.abs(observations - predictions))


def _samples_to_proba(X_test, y_test):
    """
    For every class label in y_test, it works out its actual percentage frequency for each row in X_test.
    This essentially mimics scikit's predict_proba method, except this one works on observed probabilities.

    :param X_test: Array-like object of training data
    :param y_test: Array-like object of labels for the training data
    :return: Two-dimensional numpy array of probabilities
    """
    counts = {}

    for x, y in zip(X_test, y_test):
        x = tuple(x)

        if x not in counts:
            counts[x] = {
                'total': 0.0,
                'classes': {}
            }

        if y not in counts[x]['classes']:
            counts[x]['classes'][y] = 0

        counts[x]['total'] += 1
        counts[x]['classes'][y] += 1

    # insert a zero count for missing classes
    unique_classes = np.unique(y_test)
    for count in counts.values():
        for class_label in unique_classes:
            if class_label not in count['classes']:
                count['classes'][class_label] = 0

    data = []
    for x in X_test:
        row = counts[tuple(x)]

        data.append([count / row['total'] for count in row['classes'].values()])

    return np.array(data)
