# Simulation guide

Synthetic streams are useful for analytical and adversarial tests. Construct a
`SignalStream`, wrap it in `PhysiologicalWorld`, apply seeded perturbations,
then evaluate a declared task. Real replay uses the same interface through
the WFDB adapter, so modality-specific code stays at the adapter boundary.

The cross-layer `JointWorld` records sampling, representation, communication,
compute, resource, and downstream-value outcomes in one step. Constraints are
checked at execution time and violations remain in the result.
