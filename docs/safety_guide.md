# Safety guide

Safety is a hard execution layer. `SafetyEnvelope` and `ConstraintChecker`
classify actions as admissible, inadmissible, unsafe, or unknown. Policies may
only receive actions that pass the checker. `SafeFallbackPolicy` selects a
declared safe action, and `EmergencyStop` blocks execution after a critical
failure. Reward penalties are not a substitute for hard constraints.
