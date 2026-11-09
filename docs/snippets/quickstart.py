import xrayatlas as xa

ds = xa.datasets.get("d/abcd2345")  # current version, pinned and cached
frame = ds.spectra()  # derived channels where the dataset has them
frame.to_csv("abcd2345.csv", index=False)

model = xa.models.get("m/k7m2xq4n")  # latest published version
peaks = model.peaks  # one row per stored peak
