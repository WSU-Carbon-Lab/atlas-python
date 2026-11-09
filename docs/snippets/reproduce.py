import xrayatlas as xa

xa.datasets.get("d/abcd2345")
xa.models.get("m/k7m2xq4n")
xa.lock("analysis.lock")  # pins the host, dataset versions, model versions, and checksums

xa.restore("analysis.lock")  # re-fetches exactly those revisions
