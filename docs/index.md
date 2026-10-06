# PhysioWorld documentation

## User guides

Start with [installation](installation.md) and [getting started](getting_started.md).
Then choose [CLI reference](cli.md), [simulation](simulation_guide.md),
[datasets](dataset_guide.md), [data/time](data_time_guide.md),
[communication](communication_guide.md), [resource models](resource_guide.md),
[policies](policy_guide.md), [safety](safety_guide.md),
[clinical workloads](clinical_workloads.md), or [reproducibility](reproducibility.md).
For the native Level 3 benchmark, see [corpus usage](corpus_guide.md) and the
[corpus and experiment registry](corpus_registry.md).

## Developer guides

- [Architecture and dependency boundaries](architecture.md)
- [Public API reference](api_reference.md)
- [Extending scientific components](developer_guide.md)
- [Versioned scientific contracts](contracts.md)
- [Contribution policy](../CONTRIBUTING.md)
- [Third-party licenses and attribution](third-party.md)
- [Release tooling](../tools/README.md)

## Scientific reference

- [Scientific specification](scientific_specification.md)
- [Equation-to-implementation mapping](mathematical_reference.md)
- [Scope and evidence boundaries](scientific-scope.md)
- [Known limitations and unavailable inputs](KNOWN_LIMITATIONS.md)
- [Scientific source ledger](SCIENTIFIC_SOURCE_LEDGER.yaml)
- [Extracted reference values](REFERENCE_ARTIFACTS.yaml)
- [Test-backed claim ledger](CLAIM_LEDGER.yaml)
- [Assumption ledger](ASSUMPTION_LEDGER.yaml)
- [Theorem and numerical-reference ledger](THEOREM_LEDGER.yaml)
- [Historical research context](scientific-context.md)
- [Causal interpretation](causal_guide.md)
- [Clinical validation workflow](clinical_validation_guide.md)
- [Prior-art and interoperability context](prior_art_review.md)

## Quality and release

- [Verification Manual](quality/verification-manual.md)
- [Numerical qualification](quality/numerical-qualification.md)
- [Scientific verification](quality/scientific-verification.md)
- [Model credibility and context of use](quality/model-credibility.md)
- [Reproducibility qualification](quality/reproducibility.md)
- [Benchmark protocol](benchmark_guide.md)
- [Release qualification](quality/release-qualification.md)
- [Release artifact format and commands](../release/README.md)
- [Security policy](../SECURITY.md)

Markdown is the maintained source. `python tools/docs.py check` checks local
links and heading fragments; `python tools/docs.py build --output /path/to/new-site`
builds an offline HTML tree. Add `--external` to opt into network link checking.
