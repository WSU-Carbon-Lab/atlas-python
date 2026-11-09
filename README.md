# xrayatlas

Python client for [X-ray Atlas](https://xrayatlas.wsu.edu), the collaborative database of
Near-Edge X-ray Absorption Fine Structure (NEXAFS) spectra.

`xrayatlas` pulls NEXAFS datasets, metadata, auxiliary files, and optical models from an Atlas
deployment, and builds, publishes, shares, forks, reviews, compares, cites, and locks optical
models. The client never fits peaks, never computes optical constants, and never normalizes
spectra itself: normalization is computed by the server, and the peaks and tensor payloads you
upload come from your own analysis code.

**Documentation:** <https://xrayatlas.wsu.edu/docs/python>

## Install

```bash
pip install xrayatlas
# or
uv add xrayatlas
```

Requires Python 3.10 or newer. Dependencies: numpy, pandas, httpx, and pydantic.

## Sign in

Public datasets and public models read anonymously. Writes need a token, which the
[`xatlas` CLI](https://xrayatlas.wsu.edu/docs/cli) issues and stores in a credential file this
client shares:

```bash
xatlas login
```

`xa.login()` runs the same command from Python. The client reads
`$XDG_CONFIG_HOME/xatlas/credentials.json` (default `~/.config/xatlas/credentials.json`);
`XRAYATLAS_TOKEN` and `XRAYATLAS_HOST` override the file. A token is attached only for the host it
was issued for, and is never printed or written into the HTTP cache or a lockfile.

## Quickstart

```python
import xrayatlas as xa

ds = xa.datasets.get("d/abcd2345")  # current version, pinned and cached
frame = ds.spectra()  # pandas DataFrame of the derived channels the dataset has
geometries = ds.geometries()  # distinct (theta_deg, phi_deg) pairs; (90, 45) != (45, 90)

model = xa.models.get("m/k7m2xq4n")  # latest published version
peaks = model.peaks  # one row per stored peak
channels = model.normalized(ds)  # server-computed normalized channels for a linked dataset

xa.lock("analysis.lock")  # pin every dataset and model version this session read
```

## Guides

| Topic | What it covers |
| --- | --- |
| [Datasets](https://xrayatlas.wsu.edu/docs/python/datasets) | Load by Atlas id or UUID, version pins, search, spectra, auxiliary files, BibTeX |
| [Models](https://xrayatlas.wsu.edu/docs/python/models) | Read published models: normalization, peaks, linked datasets, lineage, tensors |
| [Build and publish](https://xrayatlas.wsu.edu/docs/python/build) | The `xa.model(...)` builder, drafts, publishing, sharing, forking, reviews |
| [Compare](https://xrayatlas.wsu.edu/docs/python/compare) | Align peaks and normalized channels across models |
| [Reproduce](https://xrayatlas.wsu.edu/docs/python/reproduce) | Lockfiles with `xa.lock` and `xa.restore` |
| [Concepts](https://xrayatlas.wsu.edu/docs/python/concepts) | Datasets, versions, models, and what the server computes |
| [Errors](https://xrayatlas.wsu.edu/docs/python/errors) | The `AtlasError` hierarchy and when each error is raised |
| [Reference](https://xrayatlas.wsu.edu/docs/python/reference) | Full API reference |

Runnable scripts for each workflow live in
[`examples/`](https://github.com/WSU-Carbon-Lab/atlas-python/tree/main/examples).

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

Releases publish to PyPI automatically when a merge to `main` changes `version` in
`pyproject.toml` (`uv version --bump patch|minor|major`); see
[`.github/workflows/release.yml`](https://github.com/WSU-Carbon-Lab/atlas-python/blob/main/.github/workflows/release.yml).

## License

BSD-3-Clause. See [LICENSE](https://github.com/WSU-Carbon-Lab/atlas-python/blob/main/LICENSE).
