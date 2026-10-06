import unittest

from physioworld.causal import BinaryTreatmentSCM
from physioworld.resources import (
    CapacitorResource,
    ConstraintChecker,
    ConverterConstraints,
    EmergencyStop,
    ResourceState,
    SafetyEnvelope,
    euler_foster_state,
)
from physioworld.resources.capacitor import FosterResource


class SafetyCausalP2Tests(unittest.TestCase):
    def test_p2_rk4_matches_independent_euler(self):
        model = FosterResource(
            0.2, 0.5, v_min=0.0, time_constants_s=(1.0,), branch_resistances_ohm=(2.0,)
        )
        state = ResourceState(2.0, internal_states=(0.0,))
        transition = model.step(state, 0.1, 1.0)
        _, branch = euler_foster_state(2.0, (0.1,) * 10000, 0.2, (1.0,), (2.0,), 1.0, 10000)
        self.assertAlmostEqual(transition.next.voltage, 1.5, places=8)
        self.assertAlmostEqual(transition.next.internal_states[0], branch[0], places=4)

    def test_hard_safety_and_emergency_stop(self):
        checker = ConstraintChecker(SafetyEnvelope(min_voltage=1.85, max_current=0.1))
        self.assertEqual(checker.check(ResourceState(1.8), "sample").status.value, "unsafe")
        stop = EmergencyStop()
        stop.trigger("terminal voltage")
        with self.assertRaises(RuntimeError):
            stop.require_running()

    def test_model_aware_safety_checks_post_transition_state(self):
        model = CapacitorResource(0.2, 8.0, v_min=1.85)
        checker = ConstraintChecker(SafetyEnvelope(resource_model=model))
        check = checker.check(ResourceState(2.65), "tx", current=0.15, duration_s=0.01)
        self.assertEqual(check.status.value, "unsafe")

    def test_converter_current_limit_is_included_in_constant_bound(self):
        model = CapacitorResource(
            1.0, converter=ConverterConstraints(maximum_current_a=0.001), v_min=1.0
        )
        self.assertLessEqual(model.constant_current_limit(3.0, 1.0), 0.001)

    def test_scm_has_known_interventional_effect(self):
        model = BinaryTreatmentSCM(treatment_effect=2.0)
        self.assertEqual(model.true_average_treatment_effect(), 2.0)
        self.assertEqual(len(model.generate(20, seed=2)), 20)


if __name__ == "__main__":
    unittest.main()
