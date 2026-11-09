import xrayatlas as xa

for summary in xa.datasets.search(molecule_id="<molecule-uuid>", page_size=50):
    dataset = summary.load()
    print(dataset.atlas_id, dataset.molecule_name, len(dataset.spectra()))
