from pathlib import Path
import re
import pandas as pd


# ============================================================
# 1. ACTUAL HPC DIRECTORIES
# ============================================================

WELL_DIRECTORIES = {
    "P-1": {
        "horizons": Path(
            "/home/apgce_team_04/link-to-apgce_team_04/p-1/horizons"
        ),
        "faults": Path(
            "/home/apgce_team_04/link-to-apgce_team_04/p-1/faults"
        ),
    },

    "NO-1": {
        "horizons": Path(
            "/home/apgce_team_04/link-to-apgce_team_04/no-1/horizon"
        ),
        "faults": Path(
            "/home/apgce_team_04/link-to-apgce_team_04/no-1/faults"
        ),
    },

    "M-1": {
        "horizons": Path(
            "/home/apgce_team_04/link-to-apgce_team_04/m-1/horizon data m1"
        ),
        "faults": Path(
            "/home/apgce_team_04/link-to-apgce_team_04/m-1"
        ),
    },
}


# Optional propagated data directory.
# Place propagated files here later if you want to load them.
BASE_DIR = Path(__file__).resolve().parent
PROPAGATED_DIR = BASE_DIR / "interpretation_data" / "propagated"


# ============================================================
# 2. HORIZON FILE PATTERNS
# ============================================================

WELL_PATTERNS = {
    "P-1": re.compile(
        r"^p-1_h(\d+)(?:\.dat)?$",
        re.IGNORECASE
    ),

    "NO-1": re.compile(
        r"^no_h(\d+)\.dat$",
        re.IGNORECASE
    ),

    "M-1": re.compile(
        r"^line-(\d+)_m1\.ilclt$",
        re.IGNORECASE
    ),
}


# ============================================================
# 3. READ INDIVIDUAL HORIZON FILES
# ============================================================

def read_numeric_dat(filepath):
    """
    Read whitespace-separated numeric horizon data.

    Expected columns:
    IL, CL, TWT (ms)
    """

    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            comment="#",
            engine="python"
        )

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="any")

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return pd.DataFrame()


def load_well_horizon(filepath, well_name, horizon_id):
    """
    Load one horizon file and assign column names.
    """

    df = read_numeric_dat(filepath)

    if df.empty:
        return pd.DataFrame()

    if df.shape[1] < 3:
        print(f"Skipping {filepath}: fewer than 3 columns.")
        return pd.DataFrame()

    df = df.iloc[:, :3].copy()
    df.columns = ["IL", "CL", "TWT_ms"]

    df["Well"] = well_name
    df["HorizonID"] = int(horizon_id)
    df["SourceFile"] = filepath.name

    return df


def discover_well_horizons():
    """
    Automatically discover all horizon files
    in the three well directories.
    """

    discovered = []

    for well_name, folders in WELL_DIRECTORIES.items():

        horizon_folder = folders["horizons"]
        pattern = WELL_PATTERNS[well_name]

        if not horizon_folder.exists():
            print(
                f"WARNING: Horizon folder not found: "
                f"{horizon_folder}"
            )
            continue

        for filepath in horizon_folder.rglob("*"):

            if not filepath.is_file():
                continue

            match = pattern.match(filepath.name)

            if match:
                horizon_id = int(match.group(1))

                discovered.append({
                    "Well": well_name,
                    "HorizonID": horizon_id,
                    "FilePath": filepath
                })

    return sorted(
        discovered,
        key=lambda x: (x["Well"], x["HorizonID"])
    )


def load_all_well_horizons():
    """
    Load all horizon files from P-1, NO-1 and M-1.

    Returns:
        {
            "P-1": DataFrame,
            "NO-1": DataFrame,
            "M-1": DataFrame,
            "ALL": combined DataFrame
        }
    """

    result = {
        "P-1": [],
        "NO-1": [],
        "M-1": [],
    }

    discovered = discover_well_horizons()

    for item in discovered:

        df = load_well_horizon(
            item["FilePath"],
            item["Well"],
            item["HorizonID"]
        )

        if not df.empty:
            result[item["Well"]].append(df)

    final_result = {}

    for well_name, dataframes in result.items():

        if dataframes:
            final_result[well_name] = pd.concat(
                dataframes,
                ignore_index=True
            )
        else:
            final_result[well_name] = pd.DataFrame(
                columns=[
                    "IL", "CL", "TWT_ms",
                    "Well", "HorizonID", "SourceFile"
                ]
            )

    non_empty = [
        df for df in final_result.values()
        if not df.empty
    ]

    if non_empty:
        final_result["ALL"] = pd.concat(
            non_empty,
            ignore_index=True
        )
    else:
        final_result["ALL"] = pd.DataFrame()

    return final_result


# ============================================================
# 4. READ FAULT CSV FILES
# ============================================================

def load_fault_data(filepath, well_name=None):
    """
    Load a fault CSV file.

    Expected columns:
    FaultID, StickID, IL, CL, TWT (ms)
    """

    try:
        df = pd.read_csv(filepath)

        # Clean column names
        df.columns = [
            str(col).strip().replace("\ufeff", "")
            for col in df.columns
        ]

        # Standardise common column names
        rename_map = {}

        for col in df.columns:
            clean = col.strip().lower()

            if clean == "faultid":
                rename_map[col] = "FaultID"

            elif clean == "stickid":
                rename_map[col] = "StickID"

            elif clean == "il":
                rename_map[col] = "IL"

            elif clean == "cl":
                rename_map[col] = "CL"

            elif "twt" in clean:
                rename_map[col] = "TWT_ms"

        df = df.rename(columns=rename_map)

        required = [
            "FaultID",
            "StickID",
            "IL",
            "CL",
            "TWT_ms"
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            print(
                f"Skipping fault file {filepath.name}. "
                f"Missing columns: {missing}"
            )
            return pd.DataFrame()

        for col in required:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(subset=required).copy()

        if well_name:
            df["Well"] = well_name

        df["SourceFile"] = filepath.name

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Could not read fault file {filepath}: {e}")
        return pd.DataFrame()


def discover_fault_csv():
    """
    Discover all fault CSV files for all three wells.

    M-1 has one fault CSV file.
    P-1 and NO-1 may have multiple fault CSV files.
    """

    discovered = []

    for well_name, folders in WELL_DIRECTORIES.items():

        fault_folder = folders["faults"]

        if not fault_folder.exists():
            print(
                f"WARNING: Fault folder not found: "
                f"{fault_folder}"
            )
            continue

        for filepath in fault_folder.rglob("*"):

            if not filepath.is_file():
                continue

            if filepath.suffix.lower() != ".csv":
                continue

            # Avoid loading unrelated CSV files in M-1 folder.
            if well_name == "M-1":
                if filepath.name.lower() != "fault data m1.csv":
                    continue

            discovered.append({
                "Well": well_name,
                "FilePath": filepath
            })

    return discovered


def load_all_faults():
    """
    Load all fault CSV files and combine them.

    Returns:
        {
            "P-1": DataFrame,
            "NO-1": DataFrame,
            "M-1": DataFrame,
            "ALL": combined DataFrame
        }
    """

    result = {
        "P-1": [],
        "NO-1": [],
        "M-1": [],
    }

    discovered = discover_fault_csv()

    for item in discovered:

        df = load_fault_data(
            item["FilePath"],
            item["Well"]
        )

        if not df.empty:
            result[item["Well"]].append(df)

    final_result = {}

    for well_name, dataframes in result.items():

        if dataframes:
            final_result[well_name] = pd.concat(
                dataframes,
                ignore_index=True
            )
        else:
            final_result[well_name] = pd.DataFrame(
                columns=[
                    "FaultID", "StickID", "IL",
                    "CL", "TWT_ms", "Well", "SourceFile"
                ]
            )

    non_empty = [
        df for df in final_result.values()
        if not df.empty
    ]

    if non_empty:
        final_result["ALL"] = pd.concat(
            non_empty,
            ignore_index=True
        )
    else:
        final_result["ALL"] = pd.DataFrame()

    return final_result


# ============================================================
# 5. OPTIONAL PROPAGATED DATA
# ============================================================

def load_propagated_amplitude(filepath):
    """
    Read six-column propagated horizon amplitude data.

    Columns:
    IL, CL, X, Y, HorizonValue, Amplitude
    """

    df = read_numeric_dat(filepath)

    if df.empty or df.shape[1] < 6:
        return pd.DataFrame()

    df = df.iloc[:, :6].copy()

    df.columns = [
        "IL", "CL", "X", "Y",
        "HorizonValue", "Amplitude"
    ]

    return df


def load_propagated_point_grid(filepath):
    """
    Read five-column propagated horizon point grid.

    Columns:
    IL, CL, X, Y, HorizonValue
    """

    df = read_numeric_dat(filepath)

    if df.empty or df.shape[1] < 5:
        return pd.DataFrame()

    df = df.iloc[:, :5].copy()

    df.columns = [
        "IL", "CL", "X", "Y", "HorizonValue"
    ]

    return df


# ============================================================
# 6. DATASET SUMMARY
# ============================================================

def dataset_summary(horizons, faults):
    """
    Print a summary of loaded horizon and fault data.
    """

    print("\n" + "=" * 60)
    print("STRATAMIND INTERPRETATION DATA SUMMARY")
    print("=" * 60)

    print("\nHORIZON DATA")

    for well_name in ["P-1", "NO-1", "M-1"]:

        df = horizons.get(well_name, pd.DataFrame())

        if df.empty:
            print(f"{well_name}: No horizon data loaded.")
        else:
            number_of_files = df["SourceFile"].nunique()
            number_of_points = len(df)
            number_of_horizons = df["HorizonID"].nunique()

            print(
                f"{well_name}: "
                f"{number_of_files} files | "
                f"{number_of_horizons} horizons | "
                f"{number_of_points} points"
            )

    print("\nFAULT DATA")

    for well_name in ["P-1", "NO-1", "M-1"]:

        df = faults.get(well_name, pd.DataFrame())

        if df.empty:
            print(f"{well_name}: No fault data loaded.")
        else:
            number_of_files = df["SourceFile"].nunique()
            number_of_faults = df["FaultID"].nunique()
            number_of_points = len(df)

            print(
                f"{well_name}: "
                f"{number_of_files} files | "
                f"{number_of_faults} fault IDs | "
                f"{number_of_points} points"
            )

    print("\n" + "=" * 60)


# ============================================================
# 7. LOAD COMPLETE INTERPRETATION DATASET
# ============================================================

def load_interpretation_dataset():

    horizons = load_all_well_horizons()
    faults = load_all_faults()

    dataset = {
        "horizons": horizons,
        "faults": faults,
    }

    dataset_summary(horizons, faults)

    return dataset


# ============================================================
# 8. TEST THE DATA LOADER
# ============================================================

if __name__ == "__main__":

    print("\nChecking interpretation directories...")

    for well_name, folders in WELL_DIRECTORIES.items():

        print(f"\n{well_name}")

        print(
            "Horizon directory:",
            folders["horizons"],
            "| Exists:",
            folders["horizons"].exists()
        )

        print(
            "Fault directory:",
            folders["faults"],
            "| Exists:",
            folders["faults"].exists()
        )

    dataset = load_interpretation_dataset()

    print("\nData loading test completed.")