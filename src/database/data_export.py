"""Data export utilities for MIMIC4 game theory research."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from .mimic4_query import MIMIC4DataExtractor
from .patient_filter import PatientFilter

logger = logging.getLogger(__name__)


class DataExporter:
    """Export MIMIC4 patient data to various formats."""

    def __init__(self, session: Session, output_dir: str = "./output"):
        """Initialize exporter.

        Args:
            session: Database session
            output_dir: Output directory for exported files
        """
        self.session = session
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = MIMIC4DataExtractor(session)
        self.filter = PatientFilter(session)

    def export_to_json(
        self,
        data: Any,
        filename: str,
        pretty: bool = True,
    ) -> str:
        """导出数据为JSON格式.

        Args:
            data: 要导出的数据
            filename: 文件名
            pretty: 是否格式化输出

        Returns:
            导出文件的路径
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            else:
                json.dump(data, f, ensure_ascii=False, default=str)

        logger.info(f"Exported JSON to {filepath}")
        return str(filepath)

    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filename: str,
    ) -> str:
        """导出数据为CSV格式.

        Args:
            data: 要导出的数据（列表of字典）
            filename: 文件名

        Returns:
            导出文件的路径
        """
        if not data:
            logger.warning("No data to export")
            return ""

        filepath = self.output_dir / filename

        # Get all keys from all dictionaries
        keys = set()
        for item in data:
            keys.update(item.keys())
        keys = sorted(keys)

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"Exported CSV to {filepath}")
        return str(filepath)

    def export_patient_dataset(
        self,
        hadm_ids: Optional[List[int]] = None,
        format: str = "json",
    ) -> str:
        """导出完整的患者数据集.

        Args:
            hadm_ids: 要导出的住院ID列表（None表示全部）
            format: 导出格式 ('json' or 'csv')

        Returns:
            导出文件路径
        """
        if hadm_ids is None:
            # Get all eligible patients
            from .mimic4_models import EligiblePatient

            results = self.session.query(EligiblePatient.hadm_id).all()
            hadm_ids = [r.hadm_id for r in results]

        total = len(hadm_ids)
        logger.info(f"Exporting {total} patients...")
        print(f"开始导出 {total} 个患者的完整数据...")

        dataset = []
        failed = 0
        for idx, hadm_id in enumerate(hadm_ids, 1):
            try:
                # 进度提示
                if idx % 50 == 0 or idx == 1 or idx == total:
                    print(f"  进度: {idx}/{total} ({idx*100//total}%)")

                patient_data = self.extractor.get_complete_patient_data(hadm_id)
                dataset.append(
                    {
                        "hadm_id": hadm_id,
                        "data": patient_data,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to export patient {hadm_id}: {e}")
                failed += 1

        if failed > 0:
            print(f"⚠️ {failed} 个患者导出失败")

        print(f"✅ 成功导出 {len(dataset)} 个患者")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patient_dataset_{timestamp}.{format}"

        if format == "json":
            return self.export_to_json(dataset, filename)
        elif format == "csv":
            # Flatten the nested structure for CSV
            flattened = []
            for item in dataset:
                flat = {"hadm_id": item["hadm_id"]}
                # Add basic info
                if item["data"]["basic_info"]:
                    flat.update(item["data"]["basic_info"][0])
                flattened.append(flat)
            return self.export_to_csv(flattened, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_game_scenarios(
        self,
        n_per_complexity: int = 100,
    ) -> Dict[str, str]:
        """导出博弈场景数据集（按复杂度分类）.

        Args:
            n_per_complexity: 每个复杂度级别的样本数

        Returns:
            各复杂度级别的导出文件路径字典
        """
        from .patient_filter import select_patients_for_game_scenarios

        # Select patients by complexity
        scenarios = select_patients_for_game_scenarios(
            self.session,
            n_low_complexity=n_per_complexity,
            n_medium_complexity=n_per_complexity,
            n_high_complexity=n_per_complexity,
        )

        exported_files = {}

        for complexity_level, patients in scenarios.items():
            # Extract full data for each patient
            hadm_ids = [hadm_id for _, hadm_id, _ in patients]
            full_data = []

            for subject_id, hadm_id, score in patients:
                patient_data = self.extractor.get_complete_patient_data(hadm_id)
                full_data.append(
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "complexity_score": score,
                        "data": patient_data,
                    }
                )

            # Export to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_scenario_{complexity_level}_{timestamp}.json"
            filepath = self.export_to_json(full_data, filename)
            exported_files[complexity_level] = filepath

        logger.info(f"Exported game scenarios: {exported_files}")
        return exported_files

    def export_summary_statistics(self) -> str:
        """导出汇总统计信息.

        Returns:
            导出文件路径
        """
        summary = self.extractor.get_eligible_patients_summary()

        # Add additional statistics
        from .mimic4_models import EligiblePatient

        # Gender distribution
        gender_dist = (
            self.session.query(
                EligiblePatient.gender,
                func.count(EligiblePatient.subject_id).label("count"),
            )
            .group_by(EligiblePatient.gender)
            .all()
        )

        summary["gender_distribution"] = {g: count for g, count in gender_dist}

        # Age distribution
        from sqlalchemy import case, func

        age_dist = (
            self.session.query(
                case(
                    (EligiblePatient.age_at_admission.between(18, 30), "18-30"),
                    (EligiblePatient.age_at_admission.between(31, 50), "31-50"),
                    (EligiblePatient.age_at_admission.between(51, 70), "51-70"),
                    (EligiblePatient.age_at_admission.between(71, 89), "71-89"),
                    else_="Other",
                ).label("age_group"),
                func.count(EligiblePatient.subject_id).label("count"),
            )
            .group_by("age_group")
            .all()
        )

        summary["age_distribution"] = {group: count for group, count in age_dist}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_statistics_{timestamp}.json"

        return self.export_to_json(summary, filename)

    def export_for_pandas(self, hadm_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """导出数据为适合pandas的格式.

        Args:
            hadm_ids: 要导出的住院ID列表

        Returns:
            包含各类数据的字典
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed")
            return {}

        if hadm_ids is None:
            from .mimic4_models import EligiblePatient

            results = self.session.query(EligiblePatient.hadm_id).all()
            hadm_ids = [r.hadm_id for r in results]

        # Collect data for each table type
        all_basic_info = []
        all_admissions = []
        all_transfers = []
        all_diagnoses = []
        all_procedures = []
        all_prescriptions = []

        for hadm_id in hadm_ids:
            all_basic_info.extend(self.extractor.get_patient_basic_info(hadm_id))
            all_admissions.extend(self.extractor.get_admission_details(hadm_id))
            all_transfers.extend(self.extractor.get_transfers(hadm_id))
            all_diagnoses.extend(self.extractor.get_diagnoses(hadm_id))
            all_procedures.extend(self.extractor.get_procedures(hadm_id))
            all_prescriptions.extend(
                self.extractor.get_prescriptions(hadm_id, limit=50)
            )

        return {
            "basic_info": pd.DataFrame(all_basic_info),
            "admissions": pd.DataFrame(all_admissions),
            "transfers": pd.DataFrame(all_transfers),
            "diagnoses": pd.DataFrame(all_diagnoses),
            "procedures": pd.DataFrame(all_procedures),
            "prescriptions": pd.DataFrame(all_prescriptions),
        }

    def export_all_formats(
        self, hadm_ids: Optional[List[int]] = None
    ) -> Dict[str, str]:
        """一次性导出所有格式的数据.

        Args:
            hadm_ids: 要导出的住院ID列表

        Returns:
            各种格式的导出文件路径字典
        """
        exported = {}

        # Summary statistics
        exported["summary"] = self.export_summary_statistics()

        # Game scenarios
        scenario_files = self.export_game_scenarios()
        exported.update(scenario_files)

        # Full dataset
        exported["dataset_json"] = self.export_patient_dataset(hadm_ids, format="json")
        exported["dataset_csv"] = self.export_patient_dataset(hadm_ids, format="csv")

        logger.info(f"Exported all formats: {exported}")
        return exported


def quick_export_sample(
    session: Session,
    n_samples: int = 100,
    output_dir: str = "./output",
) -> Dict[str, str]:
    """快速导出样本数据.

    Args:
        session: 数据库会话
        n_samples: 样本数量
        output_dir: 输出目录

    Returns:
        导出文件路径字典
    """
    exporter = DataExporter(session, output_dir)

    # Get sample patients
    filter_tool = PatientFilter(session)
    sample_patients = filter_tool.filter_by_criteria()[:n_samples]
    hadm_ids = [hadm_id for _, hadm_id in sample_patients]

    # Export
    return exporter.export_all_formats(hadm_ids)
