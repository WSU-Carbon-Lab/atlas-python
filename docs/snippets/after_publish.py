import xrayatlas as xa

published = xa.models.get("m/k7m2xq4n")
ds = xa.datasets.get("d/abcd2345")

published.set_visibility("public")  # owner only
published.share(["0000-0002-1825-0097"])
published.review(ds, "approved", note="Checked against the reference spectrum.")

with published.fork(title="My model v2") as builder:
    builder.description = "Narrower post-edge window."
published_v2 = builder.publish()
