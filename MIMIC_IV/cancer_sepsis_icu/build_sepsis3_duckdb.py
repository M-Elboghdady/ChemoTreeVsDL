from pathlib import Path
import duckdb

ROOT = Path("/home/woody/iwi5/iwi5422h/mimiciv")
HERE = Path(__file__).resolve().parent
CONCEPTS = HERE / "concepts_duckdb"

DB = Path("/home/woody/iwi5/iwi5422h/sepsis3.duckdb")
OUT = HERE / "sepsis3.csv.gz"

con = duckdb.connect(str(DB))
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='32GB'")
con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_hosp")
con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_icu")
con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived")

raw = {
    "mimiciv_hosp": [
        "labevents",
        "microbiologyevents",
        "prescriptions",
    ],
    "mimiciv_icu": [
        "icustays",
        "chartevents",
        "inputevents",
        "outputevents",
    ],
}

for schema, tables in raw.items():
    folder = "hosp" if schema == "mimiciv_hosp" else "icu"
    for table in tables:
        p = ROOT / folder / f"{table}.csv.gz"
        print(f"[VIEW] {schema}.{table}")
        con.execute(f"""
            CREATE OR REPLACE VIEW {schema}.{table} AS
            SELECT * FROM read_csv_auto(
                '{p}',
                header=true,
                compression='gzip'
            )
        """)

order = [
    "icustay_times.sql",
    "icustay_hourly.sql",
    "weight_durations.sql",

    "chemistry.sql",
    "complete_blood_count.sql",
    "enzyme.sql",
    "gcs.sql",
    "oxygen_delivery.sql",
    "ventilator_setting.sql",
    "vitalsign.sql",
    "bg.sql",

    "urine_output.sql",
    "urine_output_rate.sql",

    "ventilation.sql",

    "dopamine.sql",
    "dobutamine.sql",
    "epinephrine.sql",
    "norepinephrine.sql",

    "sofa.sql",

    "antibiotic.sql",
    "suspicion_of_infection.sql",
    "sepsis3.sql",
]

for name in order:
    matches = list(CONCEPTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"{name}: found {len(matches)} files: {matches}")

    print(f"[RUN] {name}")
    con.execute(matches[0].read_text())

print("\n[CHECK]")
print(
    con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT stay_id) AS stays,
            SUM(CASE WHEN sepsis3 THEN 1 ELSE 0 END) AS sepsis3_rows
        FROM mimiciv_derived.sepsis3
    """).fetchdf()
)

con.execute(f"""
    COPY (
        SELECT *
        FROM mimiciv_derived.sepsis3
    )
    TO '{OUT}'
    (HEADER, DELIMITER ',', COMPRESSION GZIP)
""")

print(f"\nSaved: {OUT}")
con.close()
