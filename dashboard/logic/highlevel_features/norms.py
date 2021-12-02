from enum import Enum

"""
Norms taken from https://www.sciencedirect.com/science/article/pii/S2352721816301309
"""


class AGE:
    """
    Defines the lower limit of Age to distinguish subjects to categories.
    Other constants are based on those categories.
    """

    TODDLER = 1
    PRESCHOOLER = 3
    SCHOOL_AGE = 6
    TEENAGER = 14
    YOUNG_ADULT = 18
    ADULT = 26
    OLDER_ADULT = 65


class NORM(Enum):
    APPROPRIATE = 1
    UNCERTAIN = 0
    INAPPROPRIATE = -1

    def __int__(self):
        return self.value


def sol(age, value):
    value = value / 60  # TO MINUTES
    if 0 <= value <= 30:
        return NORM.APPROPRIATE

    elif 30 < value <= 60 and age >= AGE.OLDER_ADULT:
        return NORM.UNCERTAIN
    elif 60 < value and age >= AGE.OLDER_ADULT:
        return NORM.INAPPROPRIATE

    elif 30 < value <= 45:
        return NORM.UNCERTAIN
    else:
        return NORM.INAPPROPRIATE


def sol_appropriate():
    return '0-30 mins'


def sol_uncertain(age):
    if age >= AGE.OLDER_ADULT:
        return '31-60 mins'
    else:
        return '31-45 mins'


def sol_inappropriate(age):
    if age >= AGE.OLDER_ADULT:
        return '61+ mins'
    else:
        return '46+ mins'


def awk5plus(age, value):
    if value < 2:
        return NORM.APPROPRIATE
    elif value < 3 and age >= AGE.OLDER_ADULT:
        return NORM.APPROPRIATE
    elif value < 4 and not AGE.TEENAGER <= age < AGE.YOUNG_ADULT:
        return NORM.UNCERTAIN
    elif value < 3 and AGE.TEENAGER <= age < AGE.YOUNG_ADULT:
        return NORM.UNCERTAIN
    else:
        return NORM.INAPPROPRIATE


def awk5plus_appropriate(age):
    if age >= AGE.OLDER_ADULT:
        return '0-2'
    else:
        return '0-1'


def awk5plus_uncertain(age):
    if age >= AGE.OLDER_ADULT:
        return '3'
    elif AGE.TEENAGER <= age < AGE.YOUNG_ADULT:
        return '2'
    else:
        return '2-3'


def awk5plus_inappropriate(age):
    if AGE.TEENAGER <= age < AGE.YOUNG_ADULT:
        return '3+'
    else:
        return '4+'


def waso(age, value):
    if value <= 20:
        return NORM.APPROPRIATE
    elif value <= 30 and age >= AGE.OLDER_ADULT:
        return NORM.APPROPRIATE
    elif age >= AGE.OLDER_ADULT:
        return NORM.UNCERTAIN
    elif value <= 40:
        return NORM.UNCERTAIN
    elif value <= 50 and (age < AGE.SCHOOL_AGE or AGE.TEENAGER <= age < AGE.YOUNG_ADULT):
        return NORM.UNCERTAIN
    else:
        return NORM.INAPPROPRIATE


def waso_appropriate(age):
    if age >= AGE.OLDER_ADULT:
        return '0-30 mins'
    else:
        return '0-20 mins'


def waso_uncertain(age):
    if age >= AGE.OLDER_ADULT:
        return '31+ mins'
    elif age < AGE.SCHOOL_AGE:
        return '21-50 mins'
    elif AGE.TEENAGER <= age < AGE.YOUNG_ADULT:
        return '21-50 mins'
    else:
        return '21-40 mins'


def waso_inappropriate(age):
    if age >= AGE.OLDER_ADULT:
        return '-'
    elif age < AGE.SCHOOL_AGE:
        return '51+ mins'
    elif AGE.TEENAGER <= age < AGE.YOUNG_ADULT:
        return '51+ mins'
    else:
        return '41+ mins'


def se(age, value):
    if value >= 85:
        return NORM.APPROPRIATE
    elif value >= 75:
        return NORM.UNCERTAIN
    elif value >= 65 and AGE.YOUNG_ADULT <= age < AGE.ADULT:
        return NORM.UNCERTAIN
    else:
        return NORM.INAPPROPRIATE


def se_appropriate():
    return '100-85%'


def se_uncertain(age):
    if AGE.YOUNG_ADULT <= age < AGE.ADULT:
        return '84-65%'
    else:
        return '84-75%'


def se_inappropriate(age):
    if AGE.YOUNG_ADULT <= age < AGE.ADULT:
        return '64-0%'
    else:
        return '74-0%'
