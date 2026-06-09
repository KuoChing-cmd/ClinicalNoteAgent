"""MIMIC4 database models for asymmetric information game theory research."""

from sqlalchemy import Column, String, Integer, DateTime, Float, Text, Date
from sqlalchemy.orm import relationship
from datetime import datetime

from .models import BaseModel


class Patient(BaseModel):
    """MIMIC4 Patient model."""
    
    __tablename__ = "patients"
    
    subject_id = Column(Integer, primary_key=True)
    gender = Column(String(1))
    anchor_age = Column(Integer)
    anchor_year = Column(Integer)
    anchor_year_group = Column(String(20))
    dod = Column(Date, nullable=True)  # Date of death


class Admission(BaseModel):
    """MIMIC4 Admissions model."""
    
    __tablename__ = "admissions"
    
    hadm_id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, nullable=False, index=True)
    
    admittime = Column(DateTime, nullable=False)
    dischtime = Column(DateTime)
    deathtime = Column(DateTime)
    
    admission_type = Column(String(50))
    admission_location = Column(String(100))
    discharge_location = Column(String(100))
    
    insurance = Column(String(100))
    language = Column(String(50))
    marital_status = Column(String(50))
    race = Column(String(100))
    
    hospital_expire_flag = Column(Integer, default=0)


class ICUStay(BaseModel):
    """MIMIC4 ICU Stays model."""
    
    __tablename__ = "icustays"
    
    stay_id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, nullable=False, index=True)
    hadm_id = Column(Integer, nullable=False, index=True)
    
    intime = Column(DateTime, nullable=False)
    outtime = Column(DateTime)
    
    los = Column(Float)  # Length of stay in days
    
    first_careunit = Column(String(100))
    last_careunit = Column(String(100))


class Transfer(BaseModel):
    """MIMIC4 Transfers model."""
    
    __tablename__ = "transfers"
    
    transfer_id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, nullable=False, index=True)
    hadm_id = Column(Integer, nullable=False, index=True)
    
    eventtype = Column(String(50))
    careunit = Column(String(100))
    
    intime = Column(DateTime, nullable=False)
    outtime = Column(DateTime)


class DiagnosisICD(BaseModel):
    """MIMIC4 Diagnoses ICD model."""
    
    __tablename__ = "diagnoses_icd"
    
    hadm_id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, nullable=False, index=True)
    
    seq_num = Column(Integer, primary_key=True)
    icd_code = Column(String(20), nullable=False)
    icd_version = Column(Integer, nullable=False)


class DICDDiagnosis(BaseModel):
    """MIMIC4 ICD Diagnosis dictionary."""
    
    __tablename__ = "d_icd_diagnoses"
    
    icd_code = Column(String(20), primary_key=True)
    icd_version = Column(Integer, primary_key=True)
    
    long_title = Column(Text)


class ProcedureICD(BaseModel):
    """MIMIC4 Procedures ICD model."""
    
    __tablename__ = "procedures_icd"
    
    hadm_id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, nullable=False, index=True)
    
    seq_num = Column(Integer, primary_key=True)
    icd_code = Column(String(20), nullable=False)
    icd_version = Column(Integer, nullable=False)
    
    chartdate = Column(DateTime)


class HCPCSEvent(BaseModel):
    """MIMIC4 HCPCS events model."""

    __tablename__ = "hcpcsevents"

    hadm_id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, nullable=False, index=True)

    seq_num = Column(Integer, primary_key=True)
    hcpcs_cd = Column(String(20), nullable=False)
    hcpcs_version = Column(Integer, nullable=False)

    chartdate = Column(DateTime)


class DICDProcedure(BaseModel):
    """MIMIC4 ICD Procedure dictionary."""
    
    __tablename__ = "d_icd_procedures"
    
    icd_code = Column(String(20), primary_key=True)
    icd_version = Column(Integer, primary_key=True)
    
    long_title = Column(Text)


class Prescription(BaseModel):
    """MIMIC4 Prescriptions model."""
    
    __tablename__ = "prescriptions"
    
    subject_id = Column(Integer, primary_key=True, index=True)
    hadm_id = Column(Integer, primary_key=True, index=True)
    pharmacy_id = Column(Integer, primary_key=True)
    
    starttime = Column(DateTime)
    stoptime = Column(DateTime)
    
    drug_type = Column(String(50))
    drug = Column(String(255))
    
    dose_val_rx = Column(String(100))
    dose_unit_rx = Column(String(50))
    
    route = Column(String(50))
    
    
class EligiblePatient(BaseModel):
    """临时表：符合非对称信息博弈研究条件的患者样本."""
    
    __tablename__ = "eligible_patients"
    
    subject_id = Column(Integer, nullable=False, index=True)
    hadm_id = Column(Integer, primary_key=True)
    
    gender = Column(String(1))
    age_at_admission = Column(Integer)
    
    admittime = Column(DateTime)
    dischtime = Column(DateTime)
    los_days = Column(Integer)
    
    admission_type = Column(String(50))
    hospital_expire_flag = Column(Integer, default=0)
    
    transfer_count = Column(Integer, default=0)
    diagnosis_count = Column(Integer, default=0)
