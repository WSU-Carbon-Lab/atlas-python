# xrayatlas — Python client for X-ray Atlas

The `xrayatlas` package pulls NEXAFS datasets, metadata, auxiliary files, and optical models from
an X-ray Atlas deployment, and builds, publishes, shares, forks, reviews, compares, cites, and
locks optical models. The client never fits peaks, never computes optical constants, and never
normalizes spectra itself: normalization is always computed by the server, and the peaks and
tensor payloads you send come from your own analysis code.

## Install and login

```bash
uv pip install xrayatlas   # from an Atlas checkout; not on PyPI yet
```

The package lives in the `extensions/python` submodule of the Atlas repository. It requires Python
3.10+ and depends on numpy, pandas, httpx, and pydantic.

Login with the Atlas CLI, which writes a credential file shared with this client:

```bash
xatlas login
```

The client reads `$XDG_CONFIG_HOME/xatlas/credentials.json` (default
`~/.config/xatlas/credentials.json`). The environment variables `XRAYATLAS_TOKEN` and
`XRAYATLAS_HOST` override the file; a token is attached only for the host it was issued for, so a
token stored for one Atlas deployment is never sent to another. The token is never printed and is
never written into the HTTP cache or a lockfile.

## Examples

Runnable scripts live in [`examples/`](examples/README.md) and cover pulling a dataset, reading a
model, building and publishing a draft, comparing models, and recording or restoring a lockfile.

## Pull a dataset

```python
import xrayatlas as xa

ds = xa.datasets.get("d/abcd2345")  # the current version, pinned and cached by version
```

References accept an Atlas id (`abcd2345`, `d/abcd2345`) or an experiment UUID. A dataset may
also be pinned explicitly:

```python
old = xa.datasets.get("d/abcd2345", "1.0.2")
```

A pinned version is served from the cache even after the server moves on; a version that is
neither current nor cached raises `ConflictError` rather than silently substituting another
version.

```python
frame = ds.spectra()  # derived channels: od, mass_absorption, beta, delta where present
raw = ds.spectra(raw=True)  # only the uploaded raw channel and I0
geometries = ds.geometries()  # distinct (theta_deg, phi_deg) pairs, sorted; (90, 45) != (45, 90)
files = ds.aux_files.download("aux/", kind="sample")  # sample-aux/ files of the dataset bundle
text = ds.bibtex()  # BibTeX record for the dataset
```

Export requests carry at most 100,000 rows (the server's cap). When the result may be incomplete,
`ds.spectra()` raises `AtlasError` instead of returning a silently partial spectrum; pass
`allow_truncated=True` to accept the capped export. The dataset bundle endpoint carries the full
spectrum, but this client's `aux_files.download` extracts only the auxiliary files
(`experiment-aux/` and `sample-aux/`), so a capped dataset has no full-spectrum download yet.
Searching lists lightweight summaries; spectra load only on request:

```python
for summary in xa.datasets.search(molecule_id="<molecule-uuid>"):
    frame = summary.load().spectra()
```

## Read models

```python
import xrayatlas as xa

m = xa.models.get("m/k7m2xq4n")  # the latest published version
v2 = xa.models.get("m/k7m2xq4n", 2)  # a specific published version
```

```python
peaks = m.peaks  # pandas DataFrame, one row per peak in stored order
norm = m.normalization  # the version's normalization (server-defined)
links = m.datasets  # linked datasets, each pinned to a dataset version
source = links[0].load()  # that dataset at exactly the pinned version
raw = links[0].spectra(raw=True)  # the raw channel of the linked dataset
lineage = m.lineage()  # visible ancestors of this version, newest first
channels = m.normalized(ds)  # server-computed normalized channels for a linked dataset
bibtex = m.cite()  # BibTeX of this published version
```

`m.tensor.load()` resolves a fresh signed URL, verifies the SHA-256 digest and size on every
read, and returns a NumPy array with dtype `float64` or `complex128` (a narrow `float16`,
`float32`, or `complex64` payload published by another client widens losslessly on load). Read that
array as named parameter columns, not as a physical tensor: axis 1 follows the stored `components`,
and the values you work with are `delta` and `beta` as separate series on the shared energy axis. A
complex payload carries delta in the real part and beta in the imaginary part as a storage
convenience. A full physics tensor is future work. `m.fork()` applies to published versions —
forking a draft raises `ValidationError`.

## Build and publish

The `xa.model(...)` builder reads its source (the first dataset's default model, or the model
named by `from_model=<tag>`) on entry and writes only on a clean exit:

```python
import numpy as np
import pandas as pd
import xrayatlas as xa

ds = xa.datasets.get("d/abcd2345")
energy = np.linspace(280.0, 340.0, 61)  # strictly increasing eV grid, >= 2 points
delta = np.linspace(0.0010, 0.0030, 61)  # delta parameter column
beta = np.linspace(0.0000, 0.0020, 61)  # beta parameter column
values = np.stack([delta, beta], axis=1)  # shape (61, 2): one column per component

with xa.model(datasets=[ds], title="My model") as builder:
    builder.normalization = xa.Normalization(pre=(280.0, 283.0), post=(325.0, 340.0))
    builder.peaks = pd.DataFrame(
        {"center_ev": [285.4, 288.2], "kind": ["pi-star", "sigma-star"], "chi2": [1.1, 1.3]}
    )
    builder.tensor = xa.Tensor.from_array(
        values,
        energy_ev=energy,
        components=["delta", "beta"],
        frame="lab",
        units="1",
    )
    builder.save()  # optional; the block also saves once on a clean exit

published = builder.publish()  # always explicit, never retried
```

* Nothing is written until the `with` block exits cleanly; an exception inside the block writes
  nothing.
* The builder saves once even if you call `save()` or `publish()` inside the block.
* `peaks` is a DataFrame validated like `xa.validate_peaks`; extra columns such as `chi2` become
  each peak's `params`.
* `xa.Tensor.from_array` accepts float or complex arrays (`float16`/`float32` widen to
  `float64`, `complex64` to `complex128`, exactly) and rejects integer arrays on purpose — cast
  them yourself with `.astype(...)`. The payload is a container for named parameter columns: store
  `delta` and `beta` as two real columns, or one complex column with delta in the real part and
  beta in the imaginary part. It is not a physical tensor with a contracted basis; that is future
  work.
* Publishing is a separate explicit step (`builder.publish()` or
  `xa.models.Model.publish()`) and is never retried automatically.

After publishing:

```python
published.set_visibility("public")  # private, shared, or public (owner only)
published.share(["0000-0002-1825-0097"])  # share with ORCID iDs (owner only)
published.unshare("0000-0002-1825-0097")
published.review(ds, "approved", note="Checked against the reference spectrum.")
```

Forking a published version returns a new builder prefilled from exactly that version:

```python
with published.fork(title="My model v2") as builder:
    builder.peaks = pd.DataFrame({"center_ev": [285.4, 288.2]})
    builder.save()  # optional; saves once on a clean exit

published_v2 = builder.publish()  # publish the forked draft as its next version
```

An unpublished draft (for example one a builder saved) can still be removed:

```python
with xa.model(datasets=[ds]) as scratch:
    scratch.peaks = pd.DataFrame({"center_ev": [285.4]})
    scratch.save()  # a draft that is never published

scratch.model.delete()  # delete that draft (owner only; set once the builder saves)
```

## Compare models

```python
import xrayatlas as xa

m1 = xa.models.get("m/k7m2xq4n")
m2 = xa.models.get("m/p8q9r2s3")
ds = xa.datasets.get("d/abcd2345")  # a dataset both models link
comparison = xa.compare([m1, m2], tolerance_ev=0.2)  # at least two models

peaks = comparison.peaks  # aligned peak groups: center_ev, count, one column per model
frames = comparison.spectra(ds)  # normalized channels on shared axes + differences from m1
```

`peaks` is a pandas DataFrame with one row per aligned peak group (centers chain into a group
while each next center is at most the tolerance past the widest), and `spectra` aligns the
server-computed channel (`mass_absorption` by default, or `od`/`beta`) on exact keys, holding
`NaN` where a model lacks a value.

## Reproduce a session

Every dataset and published model pulled through `xa.datasets.get` and `xa.models.get` is
recorded in an in-memory session registry, and `xa.lock` serializes it:

```python
import xrayatlas as xa

xa.lock("analysis.lock")  # writes ./analysis.lock; returns the Path written
```

The lockfile pins one Atlas host, the dataset versions, and the model versions with their
checksums — no tokens, no ORCID. Dataset pins are version strings rather than content digests, so a
restore proves the revision, not byte-for-byte dataset content; model pins additionally carry the
version checksum. The registry is per process: `lock()` refuses to mix entries from two hosts, and
`from xrayatlas.lock import reset` clears it. Re-run the session later at exactly those revisions:

```python
xa.restore("analysis.lock")
```

`restore` re-fetches each pin at exactly its recorded version and re-registers what it fetched. A
restore fails with `ConflictError` when a pinned dataset version is unavailable (neither current
nor cached) or when a model's content checksum no longer matches the pinned one.

## Errors

All client errors derive from `AtlasError`:

| Error | Raised when |
| --- | --- |
| `AtlasError` | Anything unexpected: a malformed server response, an unsafe archive path, a tensor digest mismatch, a truncated export, or citing a draft. Carries only a message. |
| `ApiError` | Base of the HTTP failures; carries `status`, `code`, and `app_code`. |
| `NetworkError` | The server could not be reached (connection failure or timeout, after retries). |
| `AuthError` | HTTP 401: the token is missing, expired, or revoked. |
| `ForbiddenError` | HTTP 403: the token lacks a scope or the action is not allowed for you. |
| `NotFoundError` | HTTP 404: the object does not exist or is not visible to the caller. |
| `ConflictError` | HTTP 409: the object changed or another request won a race — a stale draft publish, a pinned version that is neither current nor cached, or a lockfile checksum mismatch. |
| `ValidationError` | HTTP 400 or 422 from the server, or input the client rejected before sending anything. |
