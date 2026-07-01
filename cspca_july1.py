#!/usr/bin/env python3
"""
Usage:
    python cspcafeaturesfin_v4.py \
        --input /gpfs/data/prostatelab/Lakshita/intern_multimodal_dataset.csv \
        --output cspca_features_v4.csv
"""
import argparse
import ast
import sys

import numpy as np
import pandas as pd

PRIOR_FLAG_COLUMNS = [
    "has_prior_biopsy",
    "prior_prostate_procedure",
]

TRUTHY_MAP = {
    "true": 1, "false": 0,
    "1.0": 1, "0.0": 0, "-1.0": -1, "2.0": 2,
    "1": 1, "0": 0, "-1": -1, "2": 2,
}

MAX_LESIONS = 3


def normalize_flag(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return float(value)
    return TRUTHY_MAP.get(str(value).strip().lower(), np.nan)


def clean_flag_column(df, col):
    """Returns (clean_value_with_-1_as_NaN, presence_flag, raw_n_twos)."""
    normalized = df[col].apply(normalize_flag)
    n_twos = int((normalized == 2).sum())
    present = (normalized != -1) & normalized.notna()
    clean = normalized.where(normalized != -1, np.nan)
    return clean, present.astype(int), n_twos


def parse_lesions(raw):
    if pd.isna(raw):
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def lesion_epe_flag(extraprostatic_extension):
    if not extraprostatic_extension:
        return 0
    text = str(extraprostatic_extension).lower()
    negating = ("no evidence", "without", "does not abut", "no epe")
    if any(phrase in text for phrase in negating):
        return 0
    if "epe" in text or "extraprostatic extension" in text:
        return 1
    return 0


def parse_zone(zone_str):
    """
    Parse a free-text zone string and return:
      (primary_zone, secondary_zone)
    
    where:
      - primary_zone: first zone keyword found ('peripheral', 'transition', 'central', 'other')
      - secondary_zone: second zone keyword if multi-zone ('peripheral', 'transition', 'central', or None)
    """
    if not zone_str:
        return 'other', None
    
    zl = str(zone_str).lower()
    
    # Find all zone keywords in order of appearance
    zones_found = []
    keywords = ['peripheral', 'transition', 'central']
    
    for keyword in keywords:
        if keyword in zl:
            zones_found.append(keyword)
    
    # If central not found, check for fibromuscular stroma
    if 'central' not in zones_found and 'fibromuscular' in zl:
        zones_found.append('central')
    
    primary = zones_found[0] if zones_found else 'other'
    secondary = zones_found[1] if len(zones_found) > 1 else None
    
    return primary, secondary


def lesion_summary_features(lesion_list):
    """Aggregate features across all lesions for this patient."""
    if not lesion_list:
        return pd.Series({
            "lesion_count": 0,
            "lesion_max_size_mm": 0,
            "lesion_has_epe": 0,
        })

    has_epe = max(lesion_epe_flag(l.get("extraprostatic_extension")) for l in lesion_list)

    def pirads_key(l):
        p = l.get("pirads")
        return p if isinstance(p, (int, float)) else -1

    dominant = max(lesion_list, key=pirads_key)
    size_mm = dominant.get("size_mm")
    max_size = max(size_mm) if isinstance(size_mm, list) and size_mm else 0

    return pd.Series({
        "lesion_count": len(lesion_list),
        "lesion_max_size_mm": max_size,
        "lesion_has_epe": has_epe,
    })


def lesion_features_detailed(lesion_list):
    """
    Per-lesion PIRADS + zone one-hot, for up to MAX_LESIONS lesions sorted
    by PIRADS descending (most severe lesion always in slot 1). Missing
    slots are padded with pirads=0. For multi-zone lesions, both zones are
    captured via primary + secondary zone columns.
    """
    def pirads_key(l):
        p = l.get("pirads")
        return p if isinstance(p, (int, float)) else -1

    sorted_lesions = sorted(lesion_list, key=pirads_key, reverse=True)
    sorted_lesions = sorted_lesions[:MAX_LESIONS]

    out = {}
    for i in range(MAX_LESIONS):
        slot = i + 1
        if i < len(sorted_lesions):
            l = sorted_lesions[i]
            p = l.get("pirads")
            pirads = float(p) if isinstance(p, (int, float)) else 0.0
            primary_zone, secondary_zone = parse_zone(l.get("zone"))
        else:
            pirads = 0.0
            primary_zone = None
            secondary_zone = None

        out[f"lesion_{slot}_pirads"] = pirads
        
        # Primary zone (one-hot: peripheral, transition, central, other)
        out[f"lesion_{slot}_zone_peripheral"] = int(primary_zone == 'peripheral')
        out[f"lesion_{slot}_zone_transition"] = int(primary_zone == 'transition')
        out[f"lesion_{slot}_zone_central"] = int(primary_zone == 'central')
        out[f"lesion_{slot}_zone_other"] = int(primary_zone == 'other')
        
        # Secondary zone (if multi-zone lesion)
        out[f"lesion_{slot}_zone_secondary_peripheral"] = int(secondary_zone == 'peripheral')
        out[f"lesion_{slot}_zone_secondary_transition"] = int(secondary_zone == 'transition')
        out[f"lesion_{slot}_zone_secondary_central"] = int(secondary_zone == 'central')

    return pd.Series(out)


def main():
    parser = argparse.ArgumentParser(description="Build csPCa feature table v4")
    parser.add_argument("--input", default="intern_multimodal_dataset.csv")
    parser.add_argument("--output", default="cspca_features_v4.csv")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: could not find file at {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} rows")

    labeled = df[df["csPCa"].notna()].copy()
    print(f"Rows with a known csPCa label: {len(labeled)}")

    # EXCLUSION CRITERIA
    n_hip_implant = (labeled["has_hip_implant"] == True).sum()
    n_prior_pca = (labeled["has_prior_pca_treatment"] == True).sum()
    print(f"Excluding {n_hip_implant} rows with has_hip_implant=True")
    print(f"Excluding {n_prior_pca} rows with has_prior_pca_treatment=True")
    
    labeled = labeled[
        (labeled["has_hip_implant"] != True) &
        (labeled["has_prior_pca_treatment"] != True)
    ].copy()
    print(f"Rows after exclusions: {len(labeled)}")

    inconsistent = labeled[labeled["has_biopsy"] == False]
    if len(inconsistent) > 0:
        print(f"NOTE: {len(inconsistent)} labeled rows have has_biopsy=False "
              f"(kept anyway, since csPCa is the source of truth) -- worth a spot check")

    for col in PRIOR_FLAG_COLUMNS:
        clean, present, n_twos = clean_flag_column(labeled, col)
        labeled[f"{col}_clean"] = clean
        labeled[f"{col}_present"] = present
        if n_twos:
            print(f"NOTE: {col} has {n_twos} rows with value 2 (meaning still "
                  f"unresolved) -- kept as-is, review before training")

    labeled["psa_log"] = np.log1p(labeled["psa"])
    labeled["psa_present"] = labeled["psa"].notna().astype(int)

    if "psa_density" not in labeled.columns:
        labeled["psa_density"] = labeled["psa"] / labeled["prostate_volume_cc"]

    labeled["prostate_volume_present"] = labeled["prostate_volume_cc"].notna().astype(int)
    labeled["intra_vesical_protrusion_present"] = labeled["intra_vesical_protrusion_cm"].notna().astype(int)

    parsed_lesions = labeled["lesions"].apply(parse_lesions)

    summary_feats = parsed_lesions.apply(lesion_summary_features)
    labeled = pd.concat([labeled, summary_feats], axis=1)

    detailed_feats = parsed_lesions.apply(lesion_features_detailed)
    labeled = pd.concat([labeled, detailed_feats], axis=1)

    # Build output column list
    lesion_detail_cols = []
    for i in range(1, MAX_LESIONS + 1):
        lesion_detail_cols.append(f"lesion_{i}_pirads")
        # Primary zones (one-hot)
        lesion_detail_cols.extend([
            f"lesion_{i}_zone_peripheral",
            f"lesion_{i}_zone_transition",
            f"lesion_{i}_zone_central",
            f"lesion_{i}_zone_other",
        ])
        # Secondary zones (one-hot, for multi-zone lesions)
        lesion_detail_cols.extend([
            f"lesion_{i}_zone_secondary_peripheral",
            f"lesion_{i}_zone_secondary_transition",
            f"lesion_{i}_zone_secondary_central",
        ])

    output_columns = [
        # ID / metadata (not model inputs)
        "AccessionNumber", "PatientID", "split",
        "csPCa",
        "MaxGradeGroup", "MaxGleasonScore",
        # features
        "maxPIRADS",
        "psa_log", "psa_present",
        "psa_density",
        "prostate_volume_cc", "prostate_volume_present",
        "intra_vesical_protrusion_cm", "intra_vesical_protrusion_present",
        "prior_prostate_procedure_clean", "prior_prostate_procedure_present",
        "lesion_count", "lesion_max_size_mm", "lesion_has_epe",
    ] + lesion_detail_cols

    out_df = labeled[output_columns]
    out_df.to_csv(args.output, index=False)

    n_model_features = len(output_columns) - 6  # minus ID/metadata cols
    print(f"\nWrote {len(out_df)} rows x {len(output_columns)} columns to {args.output}")
    print(f"  ({n_model_features} model input features + 6 ID/metadata columns)")
    print(f"csPCa balance: {out_df['csPCa'].value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    main()
