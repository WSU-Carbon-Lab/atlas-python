import xrayatlas as xa

m = xa.models.get("m/k7m2xq4n")  # latest published version
v2 = xa.models.get("m/k7m2xq4n", 2)  # one published version
draft = xa.models.get("m/k7m2xq4n", "draft")  # owner only

peaks = m.peaks  # ordinal, center_ev, kind, ..., params, fit_provenance
normalization = m.normalization  # the version's server-side normalization
source = m.datasets[0].load()  # the linked dataset at exactly its pinned version
channels = m.normalized(source)  # recomputed by the server under this version
lineage = m.lineage()  # visible ancestors, newest first
bibtex = m.cite()  # published versions only
