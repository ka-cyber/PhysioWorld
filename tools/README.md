# Public qualification tools

| Tool | Dependencies | Function |
|---|---|---|
| `docs.py check` | Standard library | Local Markdown links and heading fragments; `--external` opts into network |
| `docs.py build --output PATH` | Standard library | Escaped static HTML, navigation, tables, code and linked assets |
| `release_manifest.py --output PATH` | Standard library | Actual environment CycloneDX SBOM, manifest and checksums |
| `smoke_install.py ARTIFACTS --parent PATH` | Standard library + pip in clean venvs | Wheel/sdist installations, package resources, all required CLI help, numerical smoke and copied examples |
| `check_schemas.py` | `schema`, `yaml`, installed package | JSON Schema meta-validation, bundled-source parity and experiment compilation |
| `check_security.py` | `dev` / detect-secrets | Credential-literal scan with failing exit status and redacted findings |
| `check_licenses.py` | Standard library | Installed distribution license/notice evidence inventory; missing evidence fails |
| `check_sbom.py SBOM` | `dev` | CycloneDX JSON schema validation |
| `check_lock_exports.py` | uv 0.12.23 | Frozen pip-export version/marker/hash parity |

Build/release outputs are never silently overwritten. Supply an existing output
parent and a new destination directory/file. These tools retain meaningful exit
codes and do not convert absent evidence into successful qualification.

Typical offline docs use:

```bash
python tools/docs.py check
python tools/docs.py build --output /path/to/new-site
```

See [release engineering](../release/README.md),
[developer checks](../docs/developer_guide.md) and
[Verification Manual](../docs/quality/verification-manual.md).
