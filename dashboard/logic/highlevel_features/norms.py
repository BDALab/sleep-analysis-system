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


def se(age, value):
    if value >= 85:
        return NORM.APPROPRIATE
    elif value >= 75:
        return NORM.UNCERTAIN
    elif value >= 65 and AGE.YOUNG_ADULT <= age < AGE.ADULT:
        return NORM.UNCERTAIN
    else:
        return NORM.INAPPROPRIATE
