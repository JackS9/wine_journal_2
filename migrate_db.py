"""
Database Migration Script for Wine Journal Schema Refactor
- Moves Terroir from Vintage to Wine (FK: wine_id, unique; drops climate_summary)
- Moves Vinification from Vintage to Wine (FK: wine_id, unique)
- Adds weather_summary and predicted_profile_json to Vintage
- Adds sources_json to ResearchCorpus
- Adds history_md and certifications to Producer
- Creates DossierProposal table with index on vintage_id
- Preserves existing data, handling sibling-vintage collisions gracefully by keeping
  the most recently created record and logging any discarded rows.
"""

import sys
import os
import sqlite3
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Tuple


def get_table_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cursor.fetchall()]
    counts = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        counts[table] = cursor.fetchone()[0]
    return counts


def migrate_database(db_path: str = "wine_journal.db") -> Dict[str, Any]:
    print(f"Starting migration on: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    before_counts = get_table_counts(conn)
    print(f"Counts before migration: {before_counts}")

    # 1. Fetch existing terroirs and map to wine_id
    cursor.execute("""
        SELECT t.*, v.wine_id 
        FROM terroirs t
        JOIN vintages v ON t.vintage_id = v.id
        ORDER BY t.id ASC
    """)
    existing_terroirs = [dict(row) for row in cursor.fetchall()]

    # Group by wine_id to handle sibling collisions
    terroirs_by_wine: Dict[int, List[Dict[str, Any]]] = {}
    for t in existing_terroirs:
        w_id = t["wine_id"]
        terroirs_by_wine.setdefault(w_id, []).append(t)

    kept_terroirs: List[Dict[str, Any]] = []
    discarded_terroirs: List[Dict[str, Any]] = []

    for w_id, t_list in terroirs_by_wine.items():
        # Keep most recently created (highest ID)
        t_sorted = sorted(t_list, key=lambda x: x["id"])
        chosen = t_sorted[-1]
        kept_terroirs.append(chosen)
        if len(t_sorted) > 1:
            for discarded in t_sorted[:-1]:
                discarded_terroirs.append(discarded)

    # 2. Fetch existing vinifications and map to wine_id
    cursor.execute("""
        SELECT vinf.*, v.wine_id 
        FROM vinifications vinf
        JOIN vintages v ON vinf.vintage_id = v.id
        ORDER BY vinf.id ASC
    """)
    existing_vinifications = [dict(row) for row in cursor.fetchall()]

    vinifications_by_wine: Dict[int, List[Dict[str, Any]]] = {}
    for v in existing_vinifications:
        w_id = v["wine_id"]
        vinifications_by_wine.setdefault(w_id, []).append(v)

    kept_vinifications: List[Dict[str, Any]] = []
    discarded_vinifications: List[Dict[str, Any]] = []

    for w_id, v_list in vinifications_by_wine.items():
        v_sorted = sorted(v_list, key=lambda x: x["id"])
        chosen = v_sorted[-1]
        kept_vinifications.append(chosen)
        if len(v_sorted) > 1:
            for discarded in v_sorted[:-1]:
                discarded_vinifications.append(discarded)

    # 3. Alter existing tables to add new columns if they do not exist
    # Producers: history_md (TEXT), certifications (JSON)
    cursor.execute("PRAGMA table_info(producers);")
    prod_cols = {col["name"] for col in cursor.fetchall()}
    if "history_md" not in prod_cols:
        cursor.execute("ALTER TABLE producers ADD COLUMN history_md TEXT;")
    if "certifications" not in prod_cols:
        cursor.execute("ALTER TABLE producers ADD COLUMN certifications JSON;")

    # Vintages: weather_summary (TEXT), predicted_profile_json (JSON)
    cursor.execute("PRAGMA table_info(vintages);")
    vint_cols = {col["name"] for col in cursor.fetchall()}
    if "weather_summary" not in vint_cols:
        cursor.execute("ALTER TABLE vintages ADD COLUMN weather_summary TEXT;")
    if "predicted_profile_json" not in vint_cols:
        cursor.execute("ALTER TABLE vintages ADD COLUMN predicted_profile_json JSON;")

    # ResearchCorpora: sources_json (JSON)
    cursor.execute("PRAGMA table_info(research_corpora);")
    corp_cols = {col["name"] for col in cursor.fetchall()}
    if "sources_json" not in corp_cols:
        cursor.execute("ALTER TABLE research_corpora ADD COLUMN sources_json JSON;")

    # 4. Create dossier_proposals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dossier_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vintage_id INTEGER NOT NULL,
            proposal_json JSON NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME,
            FOREIGN KEY (vintage_id) REFERENCES vintages (id) ON DELETE CASCADE
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_dossier_proposals_vintage_id ON dossier_proposals (vintage_id);")

    # 5. Recreate terroirs table with wine_id (dropping climate_summary)
    cursor.execute("""
        CREATE TABLE terroirs_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wine_id INTEGER NOT NULL UNIQUE,
            single_vineyard_name VARCHAR(150),
            soil_types JSON,
            elevation_meters FLOAT,
            aspect VARCHAR(50),
            FOREIGN KEY (wine_id) REFERENCES wines (id) ON DELETE CASCADE
        );
    """)

    for t in kept_terroirs:
        cursor.execute("""
            INSERT INTO terroirs_new (wine_id, single_vineyard_name, soil_types, elevation_meters, aspect)
            VALUES (?, ?, ?, ?, ?)
        """, (
            t["wine_id"],
            t.get("single_vineyard_name"),
            t.get("soil_types"),
            t.get("elevation_meters"),
            t.get("aspect")
        ))

    cursor.execute("DROP TABLE terroirs;")
    cursor.execute("ALTER TABLE terroirs_new RENAME TO terroirs;")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_terroirs_wine_id ON terroirs (wine_id);")

    # 6. Recreate vinifications table with wine_id
    cursor.execute("""
        CREATE TABLE vinifications_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wine_id INTEGER NOT NULL UNIQUE,
            fermentation_vessel VARCHAR(100),
            yeast_type VARCHAR(50),
            oak_type VARCHAR(100),
            new_oak_percent FLOAT,
            aging_months INTEGER,
            malolactic_fermentation BOOLEAN,
            filtration_notes VARCHAR(255),
            FOREIGN KEY (wine_id) REFERENCES wines (id) ON DELETE CASCADE
        );
    """)

    for v in kept_vinifications:
        cursor.execute("""
            INSERT INTO vinifications_new (
                wine_id, fermentation_vessel, yeast_type, oak_type, new_oak_percent, 
                aging_months, malolactic_fermentation, filtration_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            v["wine_id"],
            v.get("fermentation_vessel"),
            v.get("yeast_type"),
            v.get("oak_type"),
            v.get("new_oak_percent"),
            v.get("aging_months"),
            v.get("malolactic_fermentation"),
            v.get("filtration_notes")
        ))

    cursor.execute("DROP TABLE vinifications;")
    cursor.execute("ALTER TABLE vinifications_new RENAME TO vinifications;")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_vinifications_wine_id ON vinifications (wine_id);")

    conn.commit()

    after_counts = get_table_counts(conn)
    conn.close()

    report = {
        "db_path": db_path,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "terroirs_found": len(existing_terroirs),
        "terroirs_migrated": len(kept_terroirs),
        "terroirs_discarded_collisions": discarded_terroirs,
        "vinifications_found": len(existing_vinifications),
        "vinifications_migrated": len(kept_vinifications),
        "vinifications_discarded_collisions": discarded_vinifications,
    }

    print("Migration complete!")
    print(f"Counts after migration: {after_counts}")
    if discarded_terroirs:
        print(f"Discarded sibling terroir rows: {discarded_terroirs}")
    if discarded_vinifications:
        print(f"Discarded sibling vinification rows: {discarded_vinifications}")

    return report


if __name__ == "__main__":
    target_db = sys.argv[1] if len(sys.argv) > 1 else "wine_journal.db"
    res = migrate_database(target_db)
    print(json.dumps(res, indent=2, default=str))
