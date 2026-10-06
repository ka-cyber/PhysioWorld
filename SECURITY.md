# Security policy

## Reporting a vulnerability

Use **Security → Report a vulnerability** on the GitHub repository hosting
your checkout when private reporting is enabled. If that facility is absent,
contact the hosting owner through their published contact channel to arrange
private disclosure. A public issue may request a private contact, but must not
contain exploit details, credentials, patient information, or restricted data.
No response-time guarantee is implied.

Include the affected version, installation method, minimal synthetic input,
impact, and reproduction steps. Report numerical/scientific correctness issues
through normal issues unless they also expose a security vulnerability.

## Supported security scope

Security fixes target the current development line and latest published
release. Version 0.1.0 is research software; medical-device authorization and
clinical safety require context-specific evidence. See the
[scientific scope](docs/scientific-scope.md).

## Trust boundaries

- Treat experiment configurations, datasets, plugins, FMUs, and archives as
  untrusted inputs. JSON and safe YAML are data formats; loading pickle or
  executing input text is outside the configuration contract.
- Python plugins execute with the user's process privileges. Run untrusted
  plugins in a separate, least-privileged process/container with restricted
  filesystem and network access. Installation or qualification is not isolation.
- Validate archive member paths and symlink targets before extraction; reject
  absolute paths and parent traversal. Constrain output paths at the calling
  service boundary. Never concatenate input into a shell command.
- Credentials belong in provider-approved local credential stores. Provenance
  must record dataset identity and access class without embedding secrets,
  patient identifiers, private endpoints, or restricted raw records.
- `OPEN`, `RESTRICTED`, and `CREDENTIALED` dataset classes retain the provider's
  license and access conditions. Public availability does not grant redistribution.

## Supply-chain controls

`uv.lock` pins resolved dependency versions and distribution hashes. CI uses
immutable action commit references, read-only default permissions, separate
core/scientific/dev environments, dependency auditing, secret scanning, Bandit,
and license-metadata review. Checks report failures rather than establishing
that every execution path is secure. The
[release workflow](release/README.md) produces a CycloneDX SBOM and hashed
manifest from the actual environment and artifacts. Action/dependency updates
must be reviewed and their upstream references verified.
