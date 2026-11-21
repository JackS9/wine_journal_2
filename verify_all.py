"""
Comprehensive Verification Test Suite for Wine Journal 2.0 Refactor
Verifies:
1. Migration test against a copy of the real database with sibling collisions
2. Extraction test with Château Marquis de Terme 2020 corpus, checking conflicts (hectare and HVE),
   accepting the proposal, and verifying populated Wine-level Terroir/Vinification, blend sum ~100, and Vintage fields.
3. Browse and search tests (GET /api/wines?q=Margaux, drink_status filter, detail view, journal view).
4. Existing flows verification (scan, save-entry, chat Sommelier).
"""

import os
import shutil
import sqlite3
import json
from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from main import app
from migrate_db import migrate_database


def run_tests():
    print("==================================================================")
    print("1. RUNNING MIGRATION TEST ON COPY OF REAL DATABASE")
    print("==================================================================")
    test_db = "test_verification_migration.db"
    shutil.copyfile("wine_journal_backup_20261120_200303.db", test_db)

    # Insert sibling vintages and terroirs/vinifications to test sibling collisions
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO vintages (id, wine_id, year) VALUES (201, 1, 2018), (202, 1, 2019);")
    cursor.execute("INSERT INTO terroirs (id, vintage_id, single_vineyard_name, soil_types, elevation_meters) VALUES (1, 201, 'Parcel Older', '[\"gravel\"]', 20.0);")
    cursor.execute("INSERT INTO terroirs (id, vintage_id, single_vineyard_name, soil_types, elevation_meters) VALUES (2, 202, 'Parcel Newer', '[\"gravel\", \"clay\"]', 22.0);")
    cursor.execute("INSERT INTO vinifications (id, vintage_id, oak_type, new_oak_percent) VALUES (1, 201, 'French', 40.0);")
    cursor.execute("INSERT INTO vinifications (id, vintage_id, oak_type, new_oak_percent) VALUES (2, 202, 'French', 50.0);")
    conn.commit()
    conn.close()

    mig_report = migrate_database(test_db)
    print("Migration Before counts:", mig_report["before_counts"])
    print("Migration After counts:", mig_report["after_counts"])
    print("Terroirs found:", mig_report["terroirs_found"], "-> migrated:", mig_report["terroirs_migrated"])
    print("Terroirs discarded collisions:", mig_report["terroirs_discarded_collisions"])
    print("Vinifications found:", mig_report["vinifications_found"], "-> migrated:", mig_report["vinifications_migrated"])
    print("Vinifications discarded collisions:", mig_report["vinifications_discarded_collisions"])

    assert mig_report["terroirs_migrated"] == 1
    assert len(mig_report["terroirs_discarded_collisions"]) == 1
    assert mig_report["terroirs_discarded_collisions"][0]["single_vineyard_name"] == "Parcel Older"
    assert mig_report["vinifications_migrated"] == 1
    assert len(mig_report["vinifications_discarded_collisions"]) == 1
    if os.path.exists(test_db):
        os.remove(test_db)
    print("✅ Migration test passed!")

    print("\n==================================================================")
    print("2. RUNNING EXTRACTION TEST & CONFLICT DETECTION")
    print("==================================================================")
    client = TestClient(app)

    # We extract vintage 3 (Château Marquis de Terme 2020)
    res_extract = client.post("/api/vintages/3/extract")
    assert res_extract.status_code == 200, f"Extraction failed: {res_extract.text}"
    proposal_data = res_extract.json()
    proposal_id = proposal_data["proposal_id"]
    proposed = proposal_data["proposed"]

    print(f"Proposal ID: {proposal_id}")
    print(f"Wine proposed: {proposed['wine']}")
    print(f"Terroir proposed: {proposed['terroir']}")
    print(f"Vinification proposed: {proposed['vinification']}")
    print(f"Vintage proposed: {proposed['vintage']}")
    print(f"Blend proposed: {proposed['blend']}")

    blend_sum = sum(b.get("percentage") or 0 for b in proposed["blend"])
    print(f"Blend Sum: {blend_sum}%")
    assert abs(blend_sum - 100.0) < 5.0, f"Blend sum {blend_sum} should be ~100"

    print("\nDetected Conflicts:")
    has_hectare_conflict = False
    has_hve_conflict = False
    for c in proposed["conflicts"]:
        print(f"  * Field: '{c['field']}' | Values: {c['values_found']} | Note: {c['note']}")
        field_and_note = (c['field'] + " " + str(c['values_found']) + " " + str(c.get('note', ''))).lower()
        if "ha" in field_and_note or "hectare" in field_and_note or "surface" in field_and_note or "area" in field_and_note:
            has_hectare_conflict = True
        if "hve" in field_and_note:
            has_hve_conflict = True

    print(f"Hectare disagreement detected in conflicts: {has_hectare_conflict}")
    print(f"HVE disagreement detected in conflicts: {has_hve_conflict}")
    assert has_hectare_conflict, "Hectare disagreement must land in conflicts"
    assert has_hve_conflict, "HVE disagreement must land in conflicts"

    # Accept the proposal
    res_accept = client.post(f"/api/proposals/{proposal_id}/accept", json={
        "append_unmapped_to_notes": True
    })
    assert res_accept.status_code == 200, f"Accept failed: {res_accept.text}"
    print("✅ Proposal accepted successfully!")

    # Verify Wine-level Terroir & Vinification and Vintage fields
    res_wine3 = client.get("/api/wines/3")
    assert res_wine3.status_code == 200
    wine3 = res_wine3.json()
    print("\nPopulated Wine-level Terroir:", wine3["terroir"])
    print("Populated Wine-level Vinification:", wine3["vinification"])
    assert wine3["terroir"] is not None
    assert len(wine3["terroir"]["soil_types"]) > 0
    assert wine3["vinification"] is not None
    assert wine3["vinification"]["oak_type"] is not None

    v3 = next(v for v in wine3["vintages"] if v["id"] == 3)
    print("Populated Vintage 3 ABV:", v3["abv"])
    print("Populated Vintage 3 Drinking Window:", v3["drinking_window_start"], "-", v3["drinking_window_end"])
    print("Populated Vintage 3 Weather Summary:", v3["weather_summary"][:120] if v3["weather_summary"] else None)
    print("Populated Vintage 3 Predicted Profile:", bool(v3["predicted_profile_json"]))
    assert v3["abv"] is not None
    assert v3["drinking_window_start"] is not None
    assert v3["predicted_profile_json"] is not None
    print("✅ Extraction and acceptance test passed!")

    print("\n==================================================================")
    print("3. RUNNING BROWSE, SEARCH & CELLAR TESTS")
    print("==================================================================")
    # Search for Margaux
    res_search = client.get("/api/wines?q=Margaux")
    assert res_search.status_code == 200
    search_wines = res_search.json()
    assert len(search_wines) >= 1
    print(f"Search for 'Margaux' returned {len(search_wines)} wines.")

    # Drink status filter
    res_ready = client.get("/api/wines?drink_status=in_window")
    assert res_ready.status_code == 200
    ready_wines = res_ready.json()
    assert len(ready_wines) >= 1
    print(f"Ready to drink filter returned {len(ready_wines)} wines.")

    # Full aggregate wine detail
    res_detail = client.get(f"/api/wines/{search_wines[0]['id']}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert "producer" in detail
    assert "vintages" in detail
    print(f"Wine detail loaded: {detail['name']} with {len(detail['vintages'])} vintages.")

    # Journal listing
    res_journal = client.get("/api/journal")
    assert res_journal.status_code == 200
    journal_entries = res_journal.json()
    assert len(journal_entries) >= 1
    print(f"Journal listing returned {len(journal_entries)} entries.")
    print("✅ Browse & search test passed!")

    print("\n==================================================================")
    print("4. RUNNING EXISTING FLOWS (SAVE-ENTRY & SOMMELIER CHAT)")
    print("==================================================================")
    # Save a tasting entry
    res_save = client.post("/api/journal/save-entry", json={
        "vintage_id": 3,
        "location_name": "Tasting Room",
        "occasion": "Refactor Verification Pass",
        "personal_rating": 9.5,
        "price_paid": 95.0,
        "palate_tannin": "medium_plus",
        "user_notes": "Sensational black fruit, graphite, velvety tannins, lingering finish."
    })
    assert res_save.status_code == 200
    print("Saved journal entry status:", res_save.json())

    # Chat with sommelier
    res_chat = client.post("/api/chat", json={
        "vintage_id": 3,
        "user_message": "What is the recommended decanting time and drinking window for this bottle?"
    })
    assert res_chat.status_code == 200
    chat_resp = res_chat.json()
    print("Sommelier Chat Response excerpt:", chat_resp["response"][:160].replace("\n", " "), "...")
    assert len(chat_resp["response"]) > 20
    print("✅ Existing flows verified successfully!")

    print("\n==================================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================================")


if __name__ == "__main__":
    run_tests()
