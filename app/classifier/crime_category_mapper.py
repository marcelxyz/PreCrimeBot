class CrimeMapper:
    THEFT = 1
    SERIOUS_CRIME = 2
    OTHER_CRIME = 3

    _categories = {
        'Bicycle theft': THEFT,
        'Other theft': THEFT,
        'Shoplifting': THEFT,
        'Theft from the person': THEFT,
        'Burglary': THEFT,
        'Vehicle crime': THEFT,

        'Robbery': SERIOUS_CRIME,
        'Possession of weapons': SERIOUS_CRIME,
        'Public disorder and weapons': SERIOUS_CRIME,
        'Criminal damage and arson': SERIOUS_CRIME,
        'Violence and sexual offences': SERIOUS_CRIME,

        'Anti-social behaviour': OTHER_CRIME,
        'Drugs': OTHER_CRIME,
        'Other crime': OTHER_CRIME,
        'Public order': OTHER_CRIME,
    }

    @staticmethod
    def map(category):
        """
        Maps a UK Police crime category to an internal category ID.

        :param category: Crime category, as defined by the UK Police
        :return: Mapped category ID
        """
        if category not in CrimeMapper._categories:
            raise RuntimeError('Category %s not found' % category)

        return CrimeMapper._categories[category]
