# Locked dependency environments

`uv.lock` is the source of truth for resolved versions, platform/Python markers
and distribution hashes. Compatible version ranges live in `pyproject.toml`;
the lock is an actual uv resolution, not a manually listed direct-dependency pin.
Use uv 0.12.23.

| File | Scope |
|---|---|
| `requirements-core.txt` | No third-party core runtime dependency |
| `requirements-scientific.txt` | Scientific, WFDB, YAML, plotting, schema and format/device extras with transitive pins/hashes |
| `requirements-dev.txt` | Development/test/build/security dependencies with transitive pins/hashes |
| `requirements-citation.txt` | Isolated cffconvert tool and its pinned/hash-verified dependencies |

```bash
python -m pip install --require-hashes -r requirements/requirements-dev.txt
python -m pip install --no-deps -e .
```

Regenerate after a reviewed dependency update:

```bash
uv lock
uv export --frozen --no-default-groups --no-emit-project --extra scientific --extra wfdb --extra yaml --extra plot --extra schema --extra arrow --extra hdf5 --extra nwb --extra dicom --extra serial --extra fmi --output-file requirements/requirements-scientific.txt --quiet
uv export --frozen --no-default-groups --no-emit-project --extra dev --output-file requirements/requirements-dev.txt --quiet
uv pip compile --python-version 3.12 --generate-hashes --output-file requirements/requirements-citation.txt requirements/citation.in --quiet
uv lock --check
```

Install the project separately from the hashed exports. Test selected extras
without introducing them into the core environment. Advisory and license checks
apply to the resolved installed versions; hashes alone are integrity evidence,
not an absence-of-vulnerability claim.

`cffconvert==2.0.0` runs as an isolated uv tool because its older jsonschema
constraint is incompatible with the modern scientific schema extra:
`uvx --python 3.12 --from cffconvert==2.0.0 --with-requirements requirements/requirements-citation.txt cffconvert --validate -i CITATION.cff`.
