# Example scripts

Run these from `extensions/python` after `uv pip install ./extensions/python` (or `uv sync`), and
log in once with `xatlas login` (the [CLI](../README.md#install-and-login) is `extensions/cli`).
Public datasets and public models also read anonymously; writes need a token.

| Script | What it shows |
| --- | --- |
| `pull_dataset.py` | Load a dataset by Atlas id or UUID, print its metadata and geometries, and export the derived or raw spectrum. |
| `read_model.py` | Read a published model, print its normalization, peaks, links, lineage, and tensor manifest, and export peaks, normalized channels, the tensor columns, and BibTeX. |
| `build_model.py` | Start from a dataset's default model (or `--from-model`), change normalization and peaks, attach a tensor from a CSV, save a draft with a clean exit, and publish on request. |
| `compare_models.py` | Align peaks across models and merge their server-normalized channels on exact geometry and energy keys. |
| `reproduce_session.py` | Record a lockfile of dataset and model versions, then restore it at exactly those revisions. |

```bash
uv run python examples/pull_dataset.py d/abcd2345 --out build/abcd2345.csv
uv run python examples/read_model.py m/k7m2xq4n --out build/model --normalized
uv run python examples/build_model.py d/abcd2345 --title "My model" --pre 280 283 --post 325 340
uv run python examples/compare_models.py m/k7m2xq4n m/p8q9r2s3 --dataset d/abcd2345
uv run python examples/reproduce_session.py d/abcd2345 --models m/k7m2xq4n --lock build/analysis.lock
```

Every script accepts `--host` to point at a non-default deployment, and errors are the client's
typed exceptions (`AuthError`, `NotFoundError`, `ConflictError`, `ValidationError`, ...). The
tensor is a container of named parameter columns such as `delta` and `beta`; it is not a physical
tensor, and a full physics tensor is future work.

The scripts are examples, not tests: they need a reachable Atlas deployment and are not run in CI.
