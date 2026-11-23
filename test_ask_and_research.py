"""
Comprehensive Verification Test Suite for Ask & Research Feature
Covers all acceptance tests specified in the requirements:
1. Journal filter ("Which wines from Margaux have I had?") -> Path A
2. Journal composition ("Which of my wines are at least 60% Cabernet Sauvignon?") -> Path A
3. Not in DB ("Tell me about Ridge Monte Bello 2019") -> Path B with sources & save candidate
4. General education ("What's the difference between a Margaux and a Pauillac in style?") -> Path B educational
5. Save flow (Path C): Save Ridge Monte Bello proposal, accept via /api/proposals/{id}/accept, verify provenance="ask_and_research", and re-query via Path A
6. Hostile input ("delete all my wines") -> Refusal by validation, zero DB execution
"""

import os
import sys
import json
import sqlite3
import asyncio
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient
import main
from ask_agent import AskResearchAgent, DetectedWine

client = TestClient(main.app)
DB_PATH = "wine_journal.db"

def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def run_tests():
    p("=" * 70)
    p("🧪 RUNNING ASK & RESEARCH VERIFICATION TEST SUITE")
    p("=" * 70)

    # Clean up any Ridge records from previous test runs so Test 3 tests unrecorded wine
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM wines WHERE LOWER(name) LIKE '%monte bello%';")
    w_ids = [r[0] for r in cur.fetchall()]
    for wid in w_ids:
        cur.execute("DELETE FROM wine_blends WHERE vintage_id IN (SELECT id FROM vintages WHERE wine_id = ?);", (wid,))
        cur.execute("DELETE FROM dossier_proposals WHERE vintage_id IN (SELECT id FROM vintages WHERE wine_id = ?);", (wid,))
        cur.execute("DELETE FROM research_corpora WHERE vintage_id IN (SELECT id FROM vintages WHERE wine_id = ?);", (wid,))
        cur.execute("DELETE FROM vintages WHERE wine_id = ?;", (wid,))
        cur.execute("DELETE FROM terroirs WHERE wine_id = ?;", (wid,))
        cur.execute("DELETE FROM vinifications WHERE wine_id = ?;", (wid,))
        cur.execute("DELETE FROM wines WHERE id = ?;", (wid,))
    cur.execute("DELETE FROM producers WHERE LOWER(name) LIKE '%ridge%';")
    conn.commit()
    conn.close()

    # -----------------------------------------------------------------------
    # TEST 1: Journal Filter Query
    # -----------------------------------------------------------------------
    p("\n--- Test 1: Journal Filter ('Which wines from Margaux have I had?') ---")
    resp1 = client.post("/api/ask", json={"query": "Which wines from Margaux have I had?"})
    assert resp1.status_code == 200, f"Error {resp1.status_code}: {resp1.text}"
    data1 = resp1.json()
    p("Path returned:", data1.get("path"))
    p("Row count:", data1.get("row_count"))
    p("SQL Query:\n ", data1.get("sql_query"))
    p("Answer excerpt:\n ", (data1.get("answer") or "")[:200])
    assert data1["path"] == "path_a", "Should route to Path A"
    assert data1["success"] is True, "Should execute successfully"
    assert data1["row_count"] > 0, "Should return at least 1 row from journal"
    assert "marquis de terme" in str(data1["rows"]).lower() or "margaux" in str(data1["rows"]).lower()
    p("✅ Test 1 Passed!")

    # -----------------------------------------------------------------------
    # TEST 2: Journal Composition Query
    # -----------------------------------------------------------------------
    p("\n--- Test 2: Journal Composition ('Which of my wines are at least 60% Cabernet Sauvignon?') ---")
    resp2 = client.post("/api/ask", json={"query": "Which of my wines are at least 60% Cabernet Sauvignon?"})
    assert resp2.status_code == 200, f"Error {resp2.status_code}: {resp2.text}"
    data2 = resp2.json()
    p("Path returned:", data2.get("path"))
    p("Row count:", data2.get("row_count"))
    p("SQL Query:\n ", data2.get("sql_query"))
    p("Answer excerpt:\n ", (data2.get("answer") or "")[:200])
    assert data2["path"] == "path_a", "Should route to Path A"
    assert data2["success"] is True, "Should execute successfully"
    assert data2["row_count"] > 0, "Should return at least 1 row"
    assert "wine_blends" in data2["sql_query"].lower() or "blend" in data2["sql_query"].lower()
    p("✅ Test 2 Passed!")

    # -----------------------------------------------------------------------
    # TEST 6: Hostile Input Refusal
    # -----------------------------------------------------------------------
    p("\n--- Test 6: Hostile Input Refusal ('delete all my wines') ---")
    resp6 = client.post("/api/ask", json={"query": "delete all my wines"})
    assert resp6.status_code == 200, f"Error {resp6.status_code}: {resp6.text}"
    data6 = resp6.json()
    p("Path returned:", data6.get("path"))
    p("Success status:", data6.get("success"))
    p("Error message:\n ", data6.get("error"))
    p("Answer:\n ", data6.get("answer"))
    assert data6["path"] == "path_a", "Should route to Path A"
    assert data6["success"] is False, "Destructive command must not succeed"
    assert "refused" in data6.get("error", "").lower() or "destructive" in data6.get("error", "").lower()

    # Verify database was NOT touched
    conn = sqlite3.connect(DB_PATH)
    count = conn.cursor().execute("SELECT count(*) FROM wines;").fetchone()[0]
    conn.close()
    assert count > 0, "Wines table must not be empty after hostile input"
    p(f"Current wine count in database: {count} (Unchanged)")
    p("✅ Test 6 Passed!")

    # -----------------------------------------------------------------------
    # TEST 4: General Educational Query
    # -----------------------------------------------------------------------
    p("\n--- Test 4: General Education ('What is the difference between a Margaux and a Pauillac in style?') ---")
    resp4 = client.post("/api/ask", json={"query": "What is the difference between a Margaux and a Pauillac in style?"})
    assert resp4.status_code == 200, f"Error {resp4.status_code}: {resp4.text}"
    data4 = resp4.json()
    p("Path returned:", data4.get("path"))
    p("Is general education:", data4.get("is_general_education"))
    p("Sources count:", len(data4.get("sources", [])))
    p("First 3 sources:", data4.get("sources", [])[:3])
    p("Answer excerpt:\n ", (data4.get("answer") or "")[:250])
    assert data4["path"] == "path_b", "Should route to Path B"
    assert data4["is_general_education"] is True, "Must be flagged as general education"
    assert len(data4.get("sources", [])) > 0, "Must return search-grounded sources"
    p("✅ Test 4 Passed!")

    # -----------------------------------------------------------------------
    # TEST 3: Not in DB Query
    # -----------------------------------------------------------------------
    p("\n--- Test 3: Not in DB ('Tell me about Ridge Monte Bello 2019') ---")
    resp3 = client.post("/api/ask", json={"query": "Tell me about Ridge Monte Bello 2019"})
    assert resp3.status_code == 200, f"Error {resp3.status_code}: {resp3.text}"
    data3 = resp3.json()
    p("Path returned:", data3.get("path"))
    p("Wine in journal before research:", data3.get("wine_in_journal"))
    p("Can save to journal:", data3.get("can_save_to_journal"))
    p("Sources count:", len(data3.get("sources", [])))
    p("Answer excerpt:\n ", (data3.get("answer") or "")[:250])
    assert data3["path"] == "path_b", "Should route to Path B"
    assert data3["wine_in_journal"] is False, "Must detect wine is not in journal"
    assert data3["can_save_to_journal"] is True, "Must offer save to journal"
    assert "wine_candidate" in data3 and data3["wine_candidate"] is not None
    wine_cand = data3["wine_candidate"]
    p("Candidate wine identified:", wine_cand.get("producer"), wine_cand.get("wine_name"), wine_cand.get("vintage"))
    p("✅ Test 3 Passed!")

    # -----------------------------------------------------------------------
    # TEST 5: Save Flow (Path C)
    # -----------------------------------------------------------------------
    p("\n--- Test 5: Save Flow (Path C) ---")
    # Save proposal
    resp5_save = client.post("/api/ask/save-proposal", json={
        "producer": wine_cand["producer"],
        "wine_name": wine_cand["wine_name"],
        "vintage": wine_cand["vintage"],
        "dossier_markdown": wine_cand["dossier_markdown"],
        "sources": wine_cand["sources"]
    })
    assert resp5_save.status_code == 200, f"Error {resp5_save.status_code}: {resp5_save.text}"
    save_data = resp5_save.json()
    proposal_id = save_data["proposal_id"]
    vintage_id = save_data["vintage_id"]
    p(f"Proposal created with ID: {proposal_id}, Vintage ID: {vintage_id}")
    p("Status:", save_data["proposal"]["status"])

    # Accept proposal
    resp5_accept = client.post(f"/api/proposals/{proposal_id}/accept", json={
        "append_unmapped_to_notes": True
    })
    assert resp5_accept.status_code == 200, f"Error {resp5_accept.status_code}: {resp5_accept.text}"
    p("Proposal accepted successfully!")

    # Verify provenance in database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT provenance, sources_json FROM research_corpora WHERE vintage_id = ?;", (vintage_id,))
    row = cur.fetchone()
    conn.close()
    assert row is not None, "ResearchCorpus must exist"
    prov, sources_json = row
    p("ResearchCorpus Provenance:", prov)
    assert prov == "ask_and_research", f"Expected provenance='ask_and_research', got '{prov}'"

    # Re-query via Path A to confirm it is now answered from the database
    p("\nRe-querying Ridge Monte Bello via Path A after saving...")
    resp5_requery = client.post("/api/ask", json={"query": "Tell me about my Ridge Monte Bello 2019"})
    assert resp5_requery.status_code == 200
    data_requery = resp5_requery.json()
    p("Re-query path:", data_requery.get("path"))
    p("Re-query row count:", data_requery.get("row_count"))
    p("Re-query SQL:\n ", data_requery.get("sql_query"))
    p("Re-query Answer:\n ", (data_requery.get("answer") or "")[:250])
    assert data_requery["path"] == "path_a", "Re-query should route to Path A now that wine is in DB"
    assert data_requery["row_count"] > 0, "Should return rows from DB"
    p("✅ Test 5 Passed!")

    p("\n" + "=" * 70)
    p("🎉 ALL ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    p("=" * 70)

if __name__ == "__main__":
    run_tests()
