from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
import json

import duckdb


@dataclass
class QualityMetrics:
    run_date: str
    input_path: str
    output_path: str | None

    rows_input: int
    rows_after_hard_rules: int
    rows_after_filters: int

    null_title: int
    null_author_name: int
    null_key: int
    duplicate_key_groups: int
    invalid_year_count: int

    pct_null_description: float
    pct_short_description: float


def _today_str() -> str:
    return date.today().isoformat()


def find_latest_silver_file(silver_dir: str = "data/silver") -> str:
    p = Path(silver_dir)
    candidates = sorted(p.glob("openlibrary_silver_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No openlibrary_silver_*.parquet found in {silver_dir}")
    return str(candidates[-1])


def run_silver_quality(
    input_parquet: str | None = None,
    output_dir: str = "data/silver",
    min_rows: int = 1000,
    min_description_len: int = 30,
    write_validated: bool = True,
) -> QualityMetrics:

    if input_parquet is None:
        input_parquet = find_latest_silver_file(output_dir)

    run_date = _today_str()
    con = duckdb.connect()

    # Inspect schema (do not assume columns exist)
    cols = set(
        c[0]
        for c in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{input_parquet}')"
        ).fetchall()
    )
    print("Columns:", sorted(cols))

    rows_input = con.sql(
        f"SELECT COUNT(*) AS n FROM read_parquet('{input_parquet}')"
    ).fetchone()[0]

    # ---- Hard rule checks (metrics) ----
    null_title = con.sql(f"""
        SELECT COUNT(*)
        FROM read_parquet('{input_parquet}')
        WHERE title IS NULL OR trim(title) = ''
    """).fetchone()[0]

    null_author_name = 0
    if "author_name" in cols:
        null_author_name = con.sql(f"""
            SELECT COUNT(*)
            FROM read_parquet('{input_parquet}')
            WHERE author_name IS NULL
        """).fetchone()[0]

    null_key = con.sql(f"""
        SELECT COUNT(*)
        FROM read_parquet('{input_parquet}')
        WHERE key IS NULL OR trim(key) = ''
    """).fetchone()[0]

    duplicate_key_groups = con.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT key, COUNT(*) c
            FROM read_parquet('{input_parquet}')
            GROUP BY key
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    # ---- Apply hard rules (build validated base) ----
    # Keep hard rules minimal & stable: key + title are required.
    validated = con.sql(f"""
        SELECT *
        FROM read_parquet('{input_parquet}')
        WHERE key IS NOT NULL AND trim(key) != ''
          AND title IS NOT NULL AND trim(title) != ''
    """)

    rows_after_hard = con.sql("SELECT COUNT(*) FROM validated").fetchone()[0]

    # ---- Quality gate ----
    if rows_after_hard < min_rows:
        raise ValueError(
            f"Quality gate failed: rows_after_hard={rows_after_hard} < min_rows={min_rows}. "
            f"Input={input_parquet}"
        )
    if duplicate_key_groups > 0:
        raise ValueError(
            f"Quality gate failed: duplicate keys exist (groups={duplicate_key_groups}). "
            f"Input={input_parquet}"
        )

    # ---- Filters for embedding quality  ----
    has_description = "description" in cols

    pct_null_description = 0.0
    pct_short_description = 0.0

    if has_description:
        pct_null_description = (
            con.sql(
                """
                SELECT AVG(
                    CASE
                        WHEN description IS NULL OR trim(description) = '' THEN 1
                        ELSE 0
                    END
                )
                FROM validated
                """
            ).fetchone()[0]
            or 0.0
        )

        pct_short_description = (
            con.sql(
                f"""
                SELECT AVG(
                    CASE
                        WHEN description IS NULL OR trim(description) = '' THEN 1
                        WHEN length(description) < {min_description_len} THEN 1
                        ELSE 0
                    END
                )
                FROM validated
                """
            ).fetchone()[0]
            or 0.0
        )

        filtered = con.sql(
            f"""
            SELECT *
            FROM validated
            WHERE description IS NOT NULL
              AND trim(description) != ''
              AND length(description) >= {min_description_len}
            """
        )
    else:
        print("[WARN] Column 'description' not found. Skip description checks & filtering.")
        filtered = con.sql("SELECT * FROM validated")

    # ---- Year checks (soft rule) ----
    year_col = None
    if "publish_year" in cols:
        year_col = "publish_year"
    elif "first_publish_year" in cols:
        year_col = "first_publish_year"
    
    invalid_year_count = 0
    
    if year_col is not None:
        invalid_year_count = con.sql(f"""
            SELECT COUNT(*)
            FROM validated
            WHERE {year_col} IS NOT NULL
              AND ({year_col} < 1400 OR {year_col} > 2027)
        """).fetchone()[0]
    else:
        print("[WARN] No year column found. Skip year checks.")

    rows_after_filters = con.sql("SELECT COUNT(*) FROM filtered").fetchone()[0]

    output_path = None
    if write_validated:
        out = Path(output_dir) / f"openlibrary_validated_{run_date}.parquet"
        con.sql(f"COPY filtered TO '{out}' (FORMAT 'parquet')")
        output_path = str(out)

    metrics = QualityMetrics(
        run_date=run_date,
        input_path=input_parquet,
        output_path=output_path,
        rows_input=rows_input,
        rows_after_hard_rules=rows_after_hard,
        rows_after_filters=rows_after_filters,
        null_title=null_title,
        null_author_name=null_author_name,
        null_key=null_key,
        duplicate_key_groups=duplicate_key_groups,
        pct_null_description=float(pct_null_description),
        pct_short_description=float(pct_short_description),
        invalid_year_count=invalid_year_count,
    )

    metrics_path = Path(output_dir) / f"quality_metrics_{run_date}.json"
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics

def run_text_quality(input_parquet: str, min_rows: int = 10, min_description_len: int = 30) -> None:
    con = duckdb.connect()

    cols = set(
        c[0]
        for c in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{input_parquet}')"
        ).fetchall()
    )
    print("Columns:", sorted(cols))

    if "key" not in cols or "description" not in cols:
        raise ValueError("Text parquet must contain columns: key, description")

    rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{input_parquet}')").fetchone()[0]
    if rows < min_rows:
        raise ValueError(f"Too few rows: {rows} < {min_rows}")

    pct_null = (
        con.sql(
            f"""
            SELECT AVG(CASE WHEN description IS NULL OR trim(description) = '' THEN 1 ELSE 0 END)
            FROM read_parquet('{input_parquet}')
            """
        ).fetchone()[0]
        or 0.0
    )

    pct_short = (
        con.sql(
            f"""
            SELECT AVG(
                CASE
                    WHEN description IS NULL OR trim(description) = '' THEN 1
                    WHEN length(description) < {min_description_len} THEN 1
                    ELSE 0
                END
            )
            FROM read_parquet('{input_parquet}')
            """
        ).fetchone()[0]
        or 0.0
    )

    print(f"rows={rows} pct_null_description={pct_null:.3f} pct_short_description={pct_short:.3f}")


if __name__ == "__main__":
    # dev run
    m = run_silver_quality(min_rows=10)
    print(m)
