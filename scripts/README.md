# Scripts

Command-line helpers should remain thin wrappers around stable package APIs.

`build_release_manifest.py` is a compatibility entry point for the maintained
[release generator](../tools/release_manifest.py). It requires the same explicit
`--output` argument and creates the same immutable evidence bundle. See
[release instructions](../release/README.md). It does not write root-level reports.
