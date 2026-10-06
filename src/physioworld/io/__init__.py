"""Configuration and interoperable record I/O."""

from .biomedical import DICOMAdapter, NWBAdapter
from .canonical import SCHEMA_VERSION, dumps, from_dict, loads, to_dict
from .clinical import BIDSAdapter, FHIRAdapter, OMOPAdapter, OpenMHealthAdapter
from .config import load_config, save_json
from .containers import ArrowAdapter, HDF5Adapter, ParquetAdapter
from .optional import MissingOptionalDependency
from .tabular import CSVAdapter, JSONAdapter, NDJSONAdapter
from .wfdb import WFDBAdapter

__all__ = [
    "load_config",
    "save_json",
    "SCHEMA_VERSION",
    "dumps",
    "loads",
    "to_dict",
    "from_dict",
    "CSVAdapter",
    "JSONAdapter",
    "NDJSONAdapter",
    "ArrowAdapter",
    "ParquetAdapter",
    "HDF5Adapter",
    "FHIRAdapter",
    "OMOPAdapter",
    "OpenMHealthAdapter",
    "BIDSAdapter",
    "NWBAdapter",
    "DICOMAdapter",
    "WFDBAdapter",
    "MissingOptionalDependency",
]
