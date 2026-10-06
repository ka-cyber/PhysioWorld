import unittest

from physioworld.causal import positivity_diagnostic
from physioworld.clinical import binary_endpoints, subgroup_metric
from physioworld.datasets import DatasetAccess, LocalFileAdapter, default_registry


class CausalAndAccessTests(unittest.TestCase):
    def test_positivity_warning(self):
        result = positivity_diagnostic((0.001, 0.5, 0.9))
        self.assertFalse(result.valid)
        self.assertEqual(result.violations, 1)

    def test_restricted_dataset_is_not_redistributable(self):
        spec = default_registry().get("mimic_waveform")
        result = LocalFileAdapter(spec).validate()
        self.assertEqual(spec.access, DatasetAccess.CREDENTIALED)
        self.assertFalse(result["redistributable"])

    def test_clinical_endpoints_and_declared_subgroups(self):
        endpoints = binary_endpoints((1, 1, 0, 0), (1, 0, 0, 0))
        self.assertAlmostEqual(endpoints["sensitivity"], 0.5)
        subgroup = subgroup_metric(
            ("a", "b", "a"),
            {"a": {"age_band": "young"}, "b": {"age_band": "older"}},
            (1.0, 0.0, 0.5),
            attribute="age_band",
        )
        self.assertEqual(subgroup.counts["young"], 2)


if __name__ == "__main__":
    unittest.main()
