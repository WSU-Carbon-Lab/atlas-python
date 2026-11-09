import xrayatlas as xa

m1 = xa.models.get("m/k7m2xq4n")
m2 = xa.models.get("m/p8q9r2s3")
comparison = xa.compare([m1, m2], tolerance_ev=0.2)

aligned = comparison.peaks  # center_ev, count, then one column per label
ds = m1.datasets[0].load()
frames = comparison.spectra(ds, channel="mass_absorption")  # differences from m1
