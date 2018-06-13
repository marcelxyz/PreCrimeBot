from geopy.geocoders import Nominatim
from dateparser import parse
import numpy as np

from classifier.location_filtering import is_within_m25
from classifier.crime_category_mapper import CrimeMapper
import classifier.file_handler as fh
import sys

sys.path.append(fh.project_root_path('app', 'classifier'))


class PredictionResult:
    def __init__(self, theft, serious, other):
        """
        :param theft: Probability of theft occurring
        :param serious: Probability of serious crime occurring
        :param other: Probability of other/minor crime occurring
        """
        self.theft = theft
        self.serious = serious
        self.other = other


def predict(date, address):
    """
    Predicts crime probabilities for a given date and adress.

    :param date: Human readable or ISO date
    :param address: Human readable address
    :return: Prediction result
    """
    parsed_address = Nominatim().geocode(address)
    parsed_date = parse(date)

    if parsed_date is None:
        raise ValueError('The date could not be parsed')

    if parsed_address is None:
        raise ValueError('The address could not be parsed')

    print("Okay, here's what I understood:")
    print(" - address:", parsed_address)
    print(" - date:", parsed_date.strftime('%Y-%m-%d'))

    if not is_within_m25(parsed_address.latitude, parsed_address.longitude):
        raise ValueError('The provided address is not within the London/M25 region')

    location_model = fh.load_model(fh.location_clusterer_path())
    crime_model = fh.load_model(fh.crime_classifier_path())

    location = location_model.predict([
        [parsed_address.latitude, parsed_address.longitude]
    ])

    X = np.array([
        [parsed_date.month, parsed_date.year, location.item(0)]
    ], dtype=np.float64)

    if crime_model.scale_data:
        X = crime_model.scaler.transform(X)

    if not crime_model.dummy_encoding:
        location_column = X[:, [2]]

        one_hot_encoded_location = crime_model.encoder.transform(location_column).toarray()

        X = np.hstack((X[:, :2], one_hot_encoded_location))

    prediction = crime_model.model.predict_proba(X)[0]

    class_mapping = {class_name: index for index, class_name in enumerate(crime_model.model.classes_)}

    return PredictionResult(
        prediction[class_mapping[CrimeMapper.THEFT]],
        prediction[class_mapping[CrimeMapper.SERIOUS_CRIME]],
        prediction[class_mapping[CrimeMapper.OTHER_CRIME]]
    )


if __name__ == '__main__':
    result = predict('tomorrow', 'westminster')
    print("theft %.2f%%" % round(result.theft * 100, 2))
    print("serious crime: %.2f%%" % round(result.serious * 100, 2))
    print("minor and other crime: %.2f%%" % round(result.other * 100, 2))
