"""Python client for X-ray Atlas NEXAFS data and optical models."""

from xrayatlas import datasets as datasets
from xrayatlas import models as models
from xrayatlas._credentials import Credential, credential_path, load_credential
from xrayatlas._http import NetworkError
from xrayatlas._login import login
from xrayatlas.client import Client, default_client, set_default_client
from xrayatlas.compare import Comparison, compare
from xrayatlas.errors import (
    ApiError,
    AtlasError,
    AuthError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from xrayatlas.lock import lock, restore
from xrayatlas.publish import model
from xrayatlas.tensor import Tensor
from xrayatlas.types import (
    Author,
    DatasetLinkIn,
    DraftContent,
    Lineage,
    LineageNode,
    ModelDetail,
    ModelSummary,
    ModelVersion,
    Normalization,
    Peak,
    TensorManifest,
    WindowPair,
)
from xrayatlas.validate import parse_model, validate_peaks

__all__ = [
    "ApiError",
    "AtlasError",
    "AuthError",
    "Author",
    "Client",
    "Comparison",
    "ConflictError",
    "Credential",
    "DatasetLinkIn",
    "DraftContent",
    "ForbiddenError",
    "Lineage",
    "LineageNode",
    "ModelDetail",
    "ModelSummary",
    "ModelVersion",
    "NetworkError",
    "Normalization",
    "NotFoundError",
    "Peak",
    "Tensor",
    "TensorManifest",
    "ValidationError",
    "WindowPair",
    "compare",
    "credential_path",
    "datasets",
    "default_client",
    "load_credential",
    "lock",
    "login",
    "model",
    "models",
    "parse_model",
    "restore",
    "set_default_client",
    "validate_peaks",
]
