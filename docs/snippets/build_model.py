import numpy as np
import pandas as pd

import xrayatlas as xa

ds = xa.datasets.get("d/abcd2345")
energy = np.linspace(280.0, 340.0, 61)  # strictly increasing, at least 2 points
delta = np.linspace(0.0010, 0.0030, 61)
beta = np.linspace(0.0000, 0.0020, 61)

with xa.model(datasets=[ds], title="My model") as builder:
    builder.normalization = xa.Normalization(pre=(280.0, 283.0), post=(325.0, 340.0))
    builder.peaks = pd.DataFrame(
        {"center_ev": [285.4, 288.2], "kind": ["pi-star", "sigma-star"], "chi2": [1.1, 1.3]}
    )
    builder.tensor = xa.Tensor.from_array(
        np.stack([delta, beta], axis=1),
        energy_ev=energy,
        components=["delta", "beta"],
        frame="lab",
        units="1",
    )

published = builder.publish()  # explicit, never retried
