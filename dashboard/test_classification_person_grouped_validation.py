import unittest

import numpy as np

from dashboard.logic.classification_person_grouped_validation import (
    _build_stratified_group_cv,
    _grouped_cv_skip_reason,
    _holdout_skip_reason,
)


class ClassificationPersonGroupedValidationTest(unittest.TestCase):
    def test_stratified_group_cv_keeps_person_groups_intact(self):
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        groups = np.array(["HC-1", "HC-1", "HC-2", "P-1", "P-1", "P-2", "HC-3", "P-3"])

        cv = _build_stratified_group_cv(y, groups, max_splits=5)

        self.assertEqual(cv.n_splits, 3)
        for train_index, test_index in cv.split(np.zeros((len(y), 1)), y, groups=groups):
            train_groups = set(groups[train_index])
            test_groups = set(groups[test_index])
            self.assertFalse(train_groups & test_groups)

    def test_grouped_cv_skip_reason_detects_too_few_positive_groups(self):
        y = np.array([0, 0, 0, 1, 1])
        groups = np.array(["HC-1", "HC-2", "HC-3", "P-1", "P-1"])

        reason = _grouped_cv_skip_reason(y, groups)

        self.assertIn("at least two person groups per class", reason)

    def test_holdout_skip_reason_uses_person_groups(self):
        y_train = np.array([0, 0, 1, 1])
        y_test = np.array([0, 1])
        groups_train = np.array(["HC-1", "HC-2", "P-1", "P-1"])

        reason = _holdout_skip_reason(y_train, y_test, groups_train)

        self.assertIn("inner grouped CV not possible", reason)


if __name__ == "__main__":
    unittest.main()
