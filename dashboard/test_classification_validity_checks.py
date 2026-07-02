import unittest

from dashboard.logic.classification_validity_checks import (
    subject_person_id,
    subject_source_cohort,
    subject_visit_index,
)


class ClassificationValidityChecksTest(unittest.TestCase):
    def test_hc_second_visit_maps_to_same_person(self):
        self.assertEqual(subject_person_id("HC2-10"), "HC-10")
        self.assertEqual(subject_person_id("HC-10"), "HC-10")
        self.assertEqual(subject_visit_index("HC2-10"), 2)
        self.assertEqual(subject_visit_index("HC-10"), 1)

    def test_predlb_second_visit_maps_to_same_person(self):
        self.assertEqual(subject_person_id("pre-LBD2-102"), "pre-LBD-102")
        self.assertEqual(subject_person_id("pre-LBD-102"), "pre-LBD-102")
        self.assertEqual(subject_visit_index("pre-LBD2-102"), 2)
        self.assertEqual(subject_visit_index("pre-LBD-102"), 1)

    def test_source_cohort_mapping(self):
        self.assertEqual(subject_source_cohort("COBEN-1087"), "COBEN")
        self.assertEqual(subject_source_cohort("HC2-10"), "HC/HC2")
        self.assertEqual(subject_source_cohort("pre-LBD2-102"), "pre-LBD/pre-LBD2")
        self.assertEqual(subject_source_cohort("MY-HC-AU5"), "MY-HC")


if __name__ == "__main__":
    unittest.main()
