# Third-party licenses and attribution

PhysioWorld is [MIT licensed](../LICENSE). Optional and development dependencies
retain their own licenses; installing PhysioWorld does not replace provider or
dataset terms. No proprietary vendor source, graphics or UI assets are licensed
by this project.

The [dependency lock](../uv.lock) records exact distributions and hashes. Major
capability dependencies include NumPy/SciPy, WFDB, PyYAML, Matplotlib and
jsonschema. License expressions, classifiers and bundled license/notice files
must be reviewed for the exact installed versions and redistribution context.

```bash
uv run --no-sync python tools/check_licenses.py --output /path/to/new-license-inventory.json
python tools/release_manifest.py --output /path/to/new-release-bundle
```

The license check fails on missing metadata/file evidence. It does not claim
legal compatibility from a license name alone. The CycloneDX SBOM records
actual installed distributions and observed license metadata; its scope includes
the interpreter environment used for generation. Generate the release inventory
in the intended runtime environment to avoid including unrelated developer tools.

When reusing open-source code, preserve upstream copyright/license/notices and
identify the affected files and version. Referencing an equation or interoperating
with a format is distinct from copying an implementation. Provider-controlled
dataset terms remain separate from software dependency licenses.
