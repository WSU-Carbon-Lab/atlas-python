import xrayatlas as xa

ds = xa.datasets.get("d/abcd2345")
old = xa.datasets.get("d/abcd2345", "1.0.2")

# Columns: energy_ev, theta_deg, phi_deg, raw_abs, i0, od, mass_absorption, beta, delta.
derived = ds.spectra()
raw = ds.spectra(raw=True)  # energy_ev, theta_deg, phi_deg, raw_abs, i0
geometries = ds.geometries()  # distinct ordered (theta_deg, phi_deg) pairs
files = ds.aux_files.download("aux/", kind="sample")
bibtex = ds.bibtex()
linked_models = ds.models(scope="all")
