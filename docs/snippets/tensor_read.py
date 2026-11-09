import xrayatlas as xa

m = xa.models.get("m/k7m2xq4n")
tensor = m.tensor
if tensor is not None and tensor.manifest is not None:
    values = tensor.load()  # float64 or complex128
    labels = tensor.manifest.components  # for example ["delta", "beta"]
    grid = tensor.manifest.energy_ev  # min, max, and count only
    print(values.shape, labels, grid)
