import xrayatlas as xa

with xa.Client(host="https://xrayatlas.wsu.edu") as client:
    ds = xa.datasets.get("d/abcd2345", client=client)
    print(ds.atlas_id)
