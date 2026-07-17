"""Database package public exports.

Keep package import lightweight. Heavy modules are loaded lazily via __getattr__
to avoid side effects when callers only need basic DB utilities.
"""

from importlib import import_module
from typing import Any

from .config import DatabaseConfig
from .connection import DatabaseManager

__all__ = [
    "DatabaseConfig",
    "DatabaseManager",
    "BaseModel",
    "IdMixin",
    "TimestampMixin",
    "CRUDOperations",
    "BatchOperations",
    "Patient",
    "Admission",
    "ICUStay",
    "Transfer",
    "DiagnosisICD",
    "DICDDiagnosis",
    "ProcedureICD",
    "DICDProcedure",
    "Prescription",
    "EligiblePatient",
    "MIMIC4DataExtractor",
    "extract_and_export_sample_data",
    "PatientFilter",
    "select_patients_for_game_scenarios",
    "DataExporter",
    "quick_export_sample",
    "EligiblePatientsPharmacyAnalyzer",
    "analyze_eligible_patients_pharmacy",
    "DatabaseLogHandler",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseModel": ("src.database.models", "BaseModel"),
    "IdMixin": ("src.database.models", "IdMixin"),
    "TimestampMixin": ("src.database.models", "TimestampMixin"),
    "CRUDOperations": ("src.database.crud", "CRUDOperations"),
    "BatchOperations": ("src.database.batch", "BatchOperations"),
    "Patient": ("src.database.mimic4_models", "Patient"),
    "Admission": ("src.database.mimic4_models", "Admission"),
    "ICUStay": ("src.database.mimic4_models", "ICUStay"),
    "Transfer": ("src.database.mimic4_models", "Transfer"),
    "DiagnosisICD": ("src.database.mimic4_models", "DiagnosisICD"),
    "DICDDiagnosis": ("src.database.mimic4_models", "DICDDiagnosis"),
    "ProcedureICD": ("src.database.mimic4_models", "ProcedureICD"),
    "DICDProcedure": ("src.database.mimic4_models", "DICDProcedure"),
    "Prescription": ("src.database.mimic4_models", "Prescription"),
    "EligiblePatient": ("src.database.mimic4_models", "EligiblePatient"),
    "MIMIC4DataExtractor": ("src.database.mimic4_query", "MIMIC4DataExtractor"),
    "extract_and_export_sample_data": (
        "src.database.mimic4_query",
        "extract_and_export_sample_data",
    ),
    "PatientFilter": ("src.database.patient_filter", "PatientFilter"),
    "select_patients_for_game_scenarios": (
        "src.database.patient_filter",
        "select_patients_for_game_scenarios",
    ),
    "DataExporter": ("src.database.data_export", "DataExporter"),
    "quick_export_sample": ("src.database.data_export", "quick_export_sample"),
    "EligiblePatientsPharmacyAnalyzer": (
        "src.database.eligible_patients_pharmacy_analyzer",
        "EligiblePatientsPharmacyAnalyzer",
    ),
    "analyze_eligible_patients_pharmacy": (
        "src.database.eligible_patients_pharmacy_analyzer",
        "analyze_eligible_patients_pharmacy",
    ),
    "DatabaseLogHandler": ("src.database.sim_log_handler", "DatabaseLogHandler"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
