#!/usr/bin/env python3
"""
Generate the CSV datasets used by the IBM Applied Data Science Capstone (SpaceX).

The script does NOT fabricate rows or metrics.

What it does:
1. Downloads the IBM course datasets used directly by later labs:
   - dataset_part_1.csv
   - Spacex.csv
   - spacex_launch_geo.csv
   - spacex_launch_dash.csv
2. Recreates dataset_part_2.csv from dataset_part_1.csv using the
   landing-label logic from the IBM data-wrangling lab.
3. Recreates dataset_part_3.csv from dataset_part_2.csv using the
   feature-engineering / one-hot encoding logic from the IBM EDA lab.
4. Scrapes the fixed Wikipedia snapshot used by the IBM web-scraping lab
   to create spacex_web_scraped.csv.

For restricted/offline environments, --source-dir can provide local copies
of the four IBM source CSVs, and --web-scrape-fallback can provide a
previously scraped copy of the SAME fixed Wikipedia snapshot.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


IBM_DATASETS = {
    "dataset_part_1.csv":
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv",
    "spacex_launch_geo.csv":
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv",
    "spacex_launch_dash.csv":
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv",
    "Spacex.csv":
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv",
}

WIKIPEDIA_SNAPSHOT = (
    "https://en.wikipedia.org/w/index.php?"
    "title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
)

EXPECTED_WEB_COLUMNS = [
    "Flight No.", "Launch site", "Payload", "Payload mass", "Orbit",
    "Customer", "Launch outcome", "Version Booster", "Booster landing",
    "Date", "Time",
]

# Names used by this ChatGPT execution cache. Normal users can simply
# omit --source-dir and the script will download the IBM files itself.
CACHE_ALIASES = {
    "dataset_part_1.csv": ["dataset_part_1.csv", "ibm_dataset_part_1.csv"],
    "spacex_launch_geo.csv": ["spacex_launch_geo.csv", "ibm_spacex_launch_geo.csv"],
    "spacex_launch_dash.csv": ["spacex_launch_dash.csv", "ibm_spacex_launch_dash.csv"],
    "Spacex.csv": ["Spacex.csv", "ibm_Spacex.csv"],
}


def download(url: str, destination: Path) -> None:
    headers = {"User-Agent": "Mozilla/5.0 IBM-Capstone-Data-Pipeline/1.0"}
    response = requests.get(url, headers=headers, timeout=45)
    response.raise_for_status()
    destination.write_bytes(response.content)


def find_cached_file(source_dir: Path, final_name: str) -> Path | None:
    for candidate in CACHE_ALIASES.get(final_name, [final_name]):
        path = source_dir / candidate
        if path.exists():
            return path
    return None


def obtain_ibm_sources(output_dir: Path, source_dir: Path | None = None) -> None:
    for filename, url in IBM_DATASETS.items():
        destination = output_dir / filename
        if source_dir is not None:
            cached = find_cached_file(source_dir, filename)
            if cached is None:
                raise FileNotFoundError(
                    f"Offline source not found for {filename} in {source_dir}"
                )
            shutil.copy2(cached, destination)
            print(f"[IBM cache] {filename}")
        else:
            download(url, destination)
            print(f"[IBM download] {filename}")


def build_dataset_part_2(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "dataset_part_1.csv")

    # IBM lab logic: 0 = unsuccessful / no recovered first stage,
    # 1 = successful first-stage landing/recovery.
    bad_outcomes = {
        "False ASDS",
        "False Ocean",
        "False RTLS",
        "None ASDS",
        "None None",
    }

    df["Class"] = [0 if outcome in bad_outcomes else 1 for outcome in df["Outcome"]]
    df.to_csv(output_dir / "dataset_part_2.csv", index=False)
    print(
        f"[derived] dataset_part_2.csv: {df.shape[0]} rows x {df.shape[1]} cols; "
        f"Class mean={df['Class'].mean():.6f}"
    )


def build_dataset_part_3(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "dataset_part_2.csv")

    feature_columns = [
        "FlightNumber",
        "PayloadMass",
        "Orbit",
        "LaunchSite",
        "Flights",
        "GridFins",
        "Reused",
        "Legs",
        "LandingPad",
        "Block",
        "ReusedCount",
        "Serial",
    ]
    features = df[feature_columns].copy()

    # Modern pandas keeps bool columns as bool when get_dummies() is called
    # without a columns list. The original IBM lab output one-hot encoded
    # GridFins/Reused/Legs as well. Listing them explicitly reproduces the
    # course dataset consistently across pandas versions.
    categorical_columns = [
        "Orbit",
        "LaunchSite",
        "LandingPad",
        "Serial",
        "GridFins",
        "Reused",
        "Legs",
    ]
    features_one_hot = pd.get_dummies(
        features, columns=categorical_columns, dtype=float
    ).astype("float64")

    features_one_hot.to_csv(output_dir / "dataset_part_3.csv", index=False)
    print(
        f"[derived] dataset_part_3.csv: "
        f"{features_one_hot.shape[0]} rows x {features_one_hot.shape[1]} cols"
    )


# Helper functions mirror the IBM web-scraping lab.
def date_time(table_cells):
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]


def booster_version(table_cells):
    return "".join(
        [
            booster
            for i, booster in enumerate(table_cells.strings)
            if i % 2 == 0
        ][0:-1]
    )


def landing_status(table_cells):
    return [item for item in table_cells.strings][0]


def get_mass(table_cells):
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        return mass[0:mass.find("kg") + 2]
    return 0


def scrape_wikipedia_snapshot() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 IBM-Capstone-Web-Scraping/1.0"}
    response = requests.get(WIKIPEDIA_SNAPSHOT, headers=headers, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    launch_dict = {
        "Flight No.": [],
        "Launch site": [],
        "Payload": [],
        "Payload mass": [],
        "Orbit": [],
        "Customer": [],
        "Launch outcome": [],
        "Version Booster": [],
        "Booster landing": [],
        "Date": [],
        "Time": [],
    }

    tables = soup.find_all("table", class_="wikitable plainrowheaders collapsible")
    if not tables:
        # More robust fallback for equivalent class representation.
        tables = [
            table for table in soup.find_all("table")
            if "wikitable" in (table.get("class") or [])
            and "plainrowheaders" in (table.get("class") or [])
        ]

    for table in tables:
        for rows in table.find_all("tr"):
            if not rows.th or not rows.th.string:
                continue

            flight_number = rows.th.string.strip()
            if not flight_number.isdigit():
                continue

            row = rows.find_all("td")
            if len(row) < 9:
                continue

            launch_dict["Flight No."].append(flight_number)

            datatimelist = date_time(row[0])
            launch_dict["Date"].append(datatimelist[0].strip(","))
            launch_dict["Time"].append(datatimelist[1])

            bv = booster_version(row[1])
            if not bv:
                if row[1].a is not None:
                    bv = row[1].a.string
                else:
                    bv = row[1].get_text(strip=True)
            launch_dict["Version Booster"].append(bv)

            launch_site = (
                row[2].a.string if row[2].a is not None and row[2].a.string
                else row[2].get_text(strip=True)
            )
            launch_dict["Launch site"].append(launch_site)

            payload = (
                row[3].a.string if row[3].a is not None and row[3].a.string
                else row[3].get_text(strip=True)
            )
            launch_dict["Payload"].append(payload)

            launch_dict["Payload mass"].append(get_mass(row[4]))

            orbit = (
                row[5].a.string if row[5].a is not None and row[5].a.string
                else row[5].get_text(strip=True)
            )
            launch_dict["Orbit"].append(orbit)

            try:
                customer = row[6].a.string
                if customer is None:
                    customer = "Various"
            except Exception:
                customer = "Various"
            launch_dict["Customer"].append(customer)

            strings = list(row[7].strings)
            launch_dict["Launch outcome"].append(strings[0] if strings else "")

            launch_dict["Booster landing"].append(landing_status(row[8]))

    df = pd.DataFrame(launch_dict)
    return df[EXPECTED_WEB_COLUMNS]


def build_web_scraped(
    output_dir: Path,
    fallback_file: Path | None = None,
) -> None:
    try:
        df = scrape_wikipedia_snapshot()
        if df.empty:
            raise RuntimeError("Wikipedia scraper returned zero rows.")
        provenance = "fresh scrape of IBM's fixed Wikipedia snapshot"
    except Exception as exc:
        if fallback_file is None:
            raise RuntimeError(
                "Wikipedia scraping failed and no fallback was supplied. "
                f"Original error: {exc}"
            ) from exc

        df = pd.read_csv(fallback_file)
        if list(df.columns) != EXPECTED_WEB_COLUMNS:
            raise ValueError(
                "The fallback web-scraped CSV does not have the IBM lab schema."
            )
        provenance = f"validated fallback of the same fixed snapshot ({fallback_file})"
        print(f"[warning] live Wikipedia scrape unavailable: {exc}")

    # The IBM lab's fixed 9-Jun-2021 snapshot yields 121 launch records.
    if len(df) != 121:
        raise ValueError(
            f"Expected 121 rows from the fixed IBM Wikipedia snapshot; got {len(df)}."
        )

    df.to_csv(output_dir / "spacex_web_scraped.csv", index=False)
    print(f"[web scrape] spacex_web_scraped.csv: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[web scrape provenance] {provenance}")


def validate_outputs(output_dir: Path) -> None:
    expected = {
        "dataset_part_1.csv": (90, 17),
        "dataset_part_2.csv": (90, 18),
        "dataset_part_3.csv": (90, 83),
        "spacex_web_scraped.csv": (121, 11),
        "spacex_launch_geo.csv": (56, 13),
        "spacex_launch_dash.csv": (56, 7),
        "Spacex.csv": (101, 10),
    }

    print("\nValidation summary")
    print("-" * 72)
    all_ok = True
    for filename, expected_shape in expected.items():
        path = output_dir / filename
        if not path.exists():
            print(f"FAIL  {filename}: missing")
            all_ok = False
            continue
        df = pd.read_csv(path)
        actual = df.shape
        status = "OK" if actual == expected_shape else "CHECK"
        if status != "OK":
            all_ok = False
        print(f"{status:5} {filename:28} {actual[0]:>3} rows x {actual[1]:>3} cols")

    # Additional reproducibility checks for derived outputs.
    part2 = pd.read_csv(output_dir / "dataset_part_2.csv")
    if abs(part2["Class"].mean() - (2 / 3)) > 1e-12:
        print("CHECK dataset_part_2.csv: Class mean differs from course snapshot.")
        all_ok = False

    part3 = pd.read_csv(output_dir / "dataset_part_3.csv")
    if not all(dtype.kind in "fiu" for dtype in part3.dtypes):
        print("CHECK dataset_part_3.csv: contains non-numeric columns.")
        all_ok = False

    print("-" * 72)
    if all_ok:
        print("All generated files match the expected IBM course snapshot shapes.")
    else:
        print("One or more files require review; no mismatch was silently ignored.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory in which to create the final CSV files.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help=(
            "Optional offline directory containing local copies of the four "
            "IBM source CSVs. If omitted, they are downloaded from IBM."
        ),
    )
    parser.add_argument(
        "--web-scrape-fallback",
        default=None,
        help=(
            "Optional previously scraped CSV from the same fixed Wikipedia "
            "snapshot. Used only if the live HTTP scrape cannot run."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_dir).resolve() if args.source_dir else None
    fallback_file = (
        Path(args.web_scrape_fallback).resolve()
        if args.web_scrape_fallback else None
    )

    obtain_ibm_sources(output_dir, source_dir=source_dir)
    build_dataset_part_2(output_dir)
    build_dataset_part_3(output_dir)
    build_web_scraped(output_dir, fallback_file=fallback_file)
    validate_outputs(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
