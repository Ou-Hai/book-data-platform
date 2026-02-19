from datetime import date
from pathlib import Path

import duckdb


def count_bronze_rows_today() -> int:
    bronze_dir = Path("data/bronze/books_raw") / f"ingestion_date={date.today()}"
    pattern = str(bronze_dir / "*.json")

    con = duckdb.connect(database=":memory:")

    query = f"""
    SELECT COUNT(*) AS n
    FROM (
        SELECT UNNEST(docs) AS doc
        FROM read_json_auto('{pattern}', maximum_object_size=104857600)
    )
    """

    n = con.execute(query).fetchone()[0]
    print("duckdb expanded rows:", n)
    return int(n)


def count_latest_per_key_today() -> int:
    bronze_dir = Path("data/bronze/books_raw") / f"ingestion_date={date.today()}"
    pattern = str(bronze_dir / "*.json")

    con = duckdb.connect(database=":memory:")

    query = f"""
    WITH expanded AS (
        SELECT UNNEST(docs) AS doc
        FROM read_json_auto('{pattern}', maximum_object_size=104857600)
    ),
    ranked AS (
        SELECT
            doc.key AS key,
            doc._ingested_at AS ingested_at,
            ROW_NUMBER() OVER (
                PARTITION BY doc.key
                ORDER BY doc._ingested_at DESC
            ) AS rn
        FROM expanded
        WHERE doc.key IS NOT NULL
    )
    SELECT COUNT(*) 
    FROM ranked
    WHERE rn = 1
    """

    n = con.execute(query).fetchone()[0]
    print("latest rows (one per key):", n)
    return int(n)