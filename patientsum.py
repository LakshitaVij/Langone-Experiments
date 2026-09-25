from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

DICOM_SUMMARY_CSV = ROOT / "dicom_summary.csv"
ALL_TAGS_CSV = ROOT / "dicom_all_tags_long.csv"

OUT_PATIENT_SUMMARY = ROOT / "patient_report_summary.csv"


def clean(x):
    """Convert NaN/None to clean string."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def unique_join(values):
    """Join unique non-empty values with semicolon."""
    vals = []
    for v in values:
        v = clean(v)
        if v and v not in vals:
            vals.append(v)
    return "; ".join(vals)


def get_possible_report_text(all_tags):
    """
    Tries to find report-like text from DICOM metadata.

    Important:
    This only works if report text is actually stored inside the DICOM metadata.
    If reports are separate .txt/.csv/.json files, we need to join those separately.
    """

    report_keywords = [
        "TextValue",
        "LongText",
        "TextString",
        "ContentDescription",
        "StudyComments",
        "AdditionalPatientHistory",
        "AdmittingDiagnosesDescription",
        "ReasonForStudy",
        "RequestedProcedureDescription",
        "PerformedProcedureStepDescription",
        "ClinicalTrialProtocolName",
        "ClinicalTrialProtocolID",
        "ClinicalTrialSiteName",
    ]

    report_names_contains = [
        "report",
        "impression",
        "findings",
        "diagnosis",
        "comment",
        "history",
        "reason",
        "description",
        "text",
        "procedure",
        "clinical",
    ]

    tags = all_tags.copy()

    for col in ["keyword", "name", "value"]:
        if col not in tags.columns:
            tags[col] = ""

    tags["keyword"] = tags["keyword"].fillna("").astype(str)
    tags["name"] = tags["name"].fillna("").astype(str)
    tags["value"] = tags["value"].fillna("").astype(str)

    mask_keyword = tags["keyword"].isin(report_keywords)

    mask_name = tags["name"].str.lower().apply(
        lambda x: any(word in x for word in report_names_contains)
    )

    possible = tags[mask_keyword | mask_name].copy()

    # Remove empty or binary junk values
    possible = possible[possible["value"].str.strip() != ""]
    possible = possible[~possible["value"].str.startswith("<bytes")]

    return possible


def make_visit_report_text(study_uid, study_df, possible_reports):
    """
    Creates one labeled visit block:
    VISIT 1
    StudyDate...
    ReportText...
    """

    study_uid = clean(study_uid)

    study_date = unique_join(study_df.get("StudyDate", []))
    study_time = unique_join(study_df.get("StudyTime", []))
    modality = unique_join(study_df.get("Modality", []))
    body_part = unique_join(study_df.get("BodyPartExamined", []))
    view_positions = unique_join(study_df.get("ViewPosition", []))
    study_descriptions = unique_join(study_df.get("StudyDescription", []))
    series_descriptions = unique_join(study_df.get("SeriesDescription", []))
    protocol_names = unique_join(study_df.get("ProtocolName", []))

    n_images = len(study_df)

    if "SeriesInstanceUID" in study_df.columns:
        n_series = study_df["SeriesInstanceUID"].replace("", pd.NA).nunique()
    else:
        n_series = ""

    report_text = ""

    if (
        possible_reports is not None
        and not possible_reports.empty
        and "StudyInstanceUID" in possible_reports.columns
    ):
        rtags = possible_reports[possible_reports["StudyInstanceUID"] == study_uid].copy()

        report_lines = []
        seen = set()

        for _, row in rtags.iterrows():
            key = clean(row.get("keyword", ""))
            name = clean(row.get("name", ""))
            value = clean(row.get("value", ""))

            if not value:
                continue

            label = key if key else name
            line = f"{label}: {value}"

            if line not in seen:
                seen.add(line)
                report_lines.append(line)

        report_text = "\n".join(report_lines)

    if not report_text:
        report_text = (
            "[No actual report text found in DICOM metadata. "
            "Using study/series/protocol descriptions only.]"
        )

    block = f"""StudyDate: {study_date}
StudyTime: {study_time}
StudyInstanceUID: {study_uid}
Modality: {modality}
BodyPartExamined: {body_part}
ViewPosition: {view_positions}
StudyDescription: {study_descriptions}
SeriesDescription: {series_descriptions}
ProtocolName: {protocol_names}
NumberOfSeries: {n_series}
NumberOfImages: {n_images}
ReportText:
{report_text}
"""

    return block


def main():
    if not DICOM_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing file: {DICOM_SUMMARY_CSV}")

    df = pd.read_csv(DICOM_SUMMARY_CSV, dtype=str).fillna("")

    print(f"Loaded: {DICOM_SUMMARY_CSV}")
    print(f"Rows/images: {len(df)}")

    if "PatientID" not in df.columns:
        raise ValueError("dicom_summary.csv must contain PatientID column.")

    if "StudyInstanceUID" not in df.columns:
        raise ValueError("dicom_summary.csv must contain StudyInstanceUID column.")

    if ALL_TAGS_CSV.exists():
        all_tags = pd.read_csv(ALL_TAGS_CSV, dtype=str).fillna("")
        possible_reports = get_possible_report_text(all_tags)
        print(f"Loaded: {ALL_TAGS_CSV}")
        print(f"Possible report-like metadata rows found: {len(possible_reports)}")
    else:
        possible_reports = pd.DataFrame()
        print("dicom_all_tags_long.csv not found, so actual report text cannot be searched.")

    patient_rows = []

    for patient_id, patient_df in df.groupby("PatientID", dropna=False):
        patient_id = clean(patient_id)

        patient_age_values = unique_join(patient_df.get("PatientAge", []))
        patient_sex_values = unique_join(patient_df.get("PatientSex", []))

        if "StudyInstanceUID" in patient_df.columns:
            num_visits = patient_df["StudyInstanceUID"].replace("", pd.NA).nunique()
        else:
            num_visits = ""

        if "SeriesInstanceUID" in patient_df.columns:
            num_series = patient_df["SeriesInstanceUID"].replace("", pd.NA).nunique()
        else:
            num_series = ""

        num_images = len(patient_df)

        visit_blocks = []

        # Sort visits by StudyDate/StudyTime if available
        sort_cols = []
        for col in ["StudyDate", "StudyTime"]:
            if col in patient_df.columns:
                sort_cols.append(col)

        if sort_cols:
            patient_df_sorted = patient_df.sort_values(sort_cols)
        else:
            patient_df_sorted = patient_df.copy()

        # One visit = one StudyInstanceUID
        grouped_visits = patient_df_sorted.groupby("StudyInstanceUID", dropna=False)

        for visit_num, (study_uid, study_df) in enumerate(grouped_visits, start=1):
            visit_text = make_visit_report_text(
                study_uid=study_uid,
                study_df=study_df,
                possible_reports=possible_reports,
            )

            labeled_visit_text = f"""VISIT {visit_num}
{visit_text}"""

            visit_blocks.append(labeled_visit_text)

        all_visits_reports_concatenated = "\n\n-----------------------------\n\n".join(visit_blocks)

        patient_rows.append({
            "PatientID": patient_id,
            "PatientAge_values": patient_age_values,
            "PatientSex_values": patient_sex_values,
            "num_visits": num_visits,
            "num_series": num_series,
            "num_images": num_images,
            "all_visits_reports_concatenated": all_visits_reports_concatenated,
        })

    out = pd.DataFrame(patient_rows)

    # Sort patients for readability
    if "PatientID" in out.columns:
        out = out.sort_values("PatientID")

    out.to_csv(OUT_PATIENT_SUMMARY, index=False)

    print()
    print("Done.")
    print(f"Wrote: {OUT_PATIENT_SUMMARY}")
    print(f"Patients: {len(out)}")


if __name__ == "__main__":
    main()