import xrayatlas as xa
from xrayatlas.publish import ModelBuilder


def set_per_channel_normalization(builder: ModelBuilder) -> None:
    builder.normalization = xa.Normalization(
        scope="per_channel",
        od=xa.WindowPair(pre=(280.0, 283.0), post=(325.0, 340.0)),
        massabsorption=xa.WindowPair(pre=(280.0, 283.0), post=(325.0, 340.0)),
        beta=xa.WindowPair(pre=(280.0, 283.0), post=(325.0, 340.0)),
    )
