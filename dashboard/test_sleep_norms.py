import unittest

from dashboard.logic.features_extraction.norms import NORM, waso, waso_seconds
from dashboard.models import SleepNight, Subject


class SleepNormsTest(unittest.TestCase):
    def test_waso_function_uses_minutes(self):
        self.assertEqual(waso(age=40, value=20), NORM.APPROPRIATE)
        self.assertEqual(waso(age=40, value=30), NORM.UNCERTAIN)
        self.assertEqual(waso(age=40, value=45), NORM.INAPPROPRIATE)

    def test_waso_seconds_converts_seconds_before_norming(self):
        self.assertEqual(waso_seconds(age=40, value=20 * 60), NORM.APPROPRIATE)
        self.assertEqual(waso_seconds(age=40, value=30 * 60), NORM.UNCERTAIN)
        self.assertEqual(waso_seconds(age=40, value=45 * 60), NORM.INAPPROPRIATE)

    def test_sleepnight_waso_norm_uses_seconds(self):
        subject = Subject(age=40)
        sleep_night = SleepNight(subject=subject, waso=30 * 60)

        self.assertEqual(sleep_night.waso_norm, NORM.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
