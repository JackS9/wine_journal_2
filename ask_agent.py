"""
Ask & Research Agent Module
Orchestrates the three-path query engine:
- Path A: Journal Database Query (runtime SQLAlchemy schema, Gemini SQL generation, strict validation, read-only execution)
- Path B: Deep Research Query (Gemini with Google Search tools for general wine education & unrecorded wines)
- Path C: Save What You Learned (creates DossierProposal for review & human-in-the-loop commit with provenance)
"""

import os
import re
import json
import sqlite3
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from sqlalchemy import create_engine, select, func, inspect

from models import (
    Base, Producer, Wine, Vintage, WineBlend, GrapeVarietal,
    Terroir, Vinification, JournalEntry, TastingProfile, ResearchCorpus,
    DossierProposal, WineType
)
from research_agent import WineResearchAgent
from dossier_extractor import DossierExtractor, extract_dossier_from_corpus


# ---------------------------------------------------------------------------
# Whitelist of Queryable Tables for Path A
# ---------------------------------------------------------------------------

ALLOWED_TABLES = {
    "producers",
    "wines",
    "vintages",
    "wine_blends",
    "grape_varietals",
    "terroirs",
    "vinifications",
    "journal_entries",
    "tasting_profiles",
    "research_corpora"
}

FORBIDDEN_SQL_PATTERNS = [
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|ATTACH|DETACH|CREATE|REPLACE|EXEC|TRUNCATE|GRANT|REVOKE)\b",
    r";\s*\S+",  # Multiple statements separated by semicolon
]


# ---------------------------------------------------------------------------
# Router Schemas
# ---------------------------------------------------------------------------

class DetectedWine(BaseModel):
    producer: Optional[str] = None
    wine_name: Optional[str] = None
    vintage: Optional[int] = None


class RoutingDecision(BaseModel):
    path: str = Field(..., description="'A' for database/journal queries, 'B' for general wine education or wines not in DB")
    reason: str = Field(..., description="Explanation of why this path was chosen")
    is_general_education: bool = Field(default=False, description="True if query is a general concept/region/educational question")
    is_specific_wine: bool = Field(default=False, description="True if query asks about a specific wine label/producer/vintage")
    wine_in_journal: bool = Field(default=False, description="True if the wine mentioned exists in the user's journal database")
    detected_wine: Optional[DetectedWine] = Field(default=None, description="Extracted producer, wine_name, vintage if applicable")


class GeneratedSQLQuery(BaseModel):
    sql: str = Field(..., description="Single SQLite SELECT query with no mutations")
    explanation: str = Field(..., description="Brief plain-language explanation of what this query fetches")


# ---------------------------------------------------------------------------
# Ask & Research Agent
# ---------------------------------------------------------------------------

class AskResearchAgent:
    """Three-path intelligent agent for querying personal journal records and conducting wine research."""

    def __init__(self, db_path: str = "wine_journal.db", api_key: Optional[str] = None):
        self.db_path = db_path
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required.")
        self.client = genai.Client(api_key=self.api_key)
        self.research_agent = WineResearchAgent(api_key=self.api_key)

    # -----------------------------------------------------------------------
    # Schema Introspection
    # -----------------------------------------------------------------------

    def get_runtime_schema_description(self) -> str:
        """Inspects SQLAlchemy declarative models at runtime to build schema prompt."""
        descriptions = []
        for table_name, table in Base.metadata.tables.items():
            if table_name not in ALLOWED_TABLES:
                continue
            cols = []
            for col in table.columns:
                fk_str = f" REFERENCES {list(col.foreign_keys)[0].target_fullname}" if col.foreign_keys else ""
                cols.append(f"  {col.name} ({col.type}){fk_str}")
            descriptions.append(f"Table: {table_name}\n" + "\n".join(cols))
        return "\n\n".join(descriptions)

    def get_journal_wines_summary(self) -> List[Dict[str, Any]]:
        """Returns concise list of wines currently in the database to assist routing."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.name AS producer_name, w.name AS wine_name, w.appellation, v.year, v.id AS vintage_id
            FROM wines w
            JOIN producers p ON w.producer_id = p.id
            LEFT JOIN vintages v ON v.wine_id = w.id
        """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {"producer": r[0], "wine": r[1], "appellation": r[2], "year": r[3], "vintage_id": r[4]}
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # 1. Routing
    # -----------------------------------------------------------------------

    def route_query(self, user_query: str) -> RoutingDecision:
        """Routes query to Path A (Journal DB) or Path B (Deep Research)."""
        journal_wines = self.get_journal_wines_summary()
        wines_context = "\n".join(
            f"- {w['producer']} - {w['wine']} ({w['year'] or 'NV'}) [Appellation: {w['appellation']}]"
            for w in journal_wines
        )

        prompt = f"""
You are the routing controller for an AI Wine Journal and Research system.
A user asked: "{user_query}"

Here are the wines currently recorded in the user's personal journal database:
{wines_context if wines_context else "(No wines currently in journal)"}

Determine the correct path:
- PATH 'A' (Journal Database Query):
  * The user asks about wines they have had, tasted, rated, logged, or have recorded in their journal (e.g., "Which wines from Margaux have I had?", "Which of my wines are at least 60% Cabernet Sauvignon?", "What wines are in my cellar?", "Show my tasting notes for Margaux").
  * Also route here if the user attempts database operations or modifications (e.g. "delete all my wines") so they can be processed by SQL validation.
  * Route here if the query specifically refers to a wine that IS in the user's journal database list above.

- PATH 'B' (Research Query):
  * The user asks a general wine education question (e.g. "What's the difference between a Margaux and a Pauillac in style?", "What does malolactic fermentation do to Chardonnay?", "What soils characterize Barolo?").
  * The user asks about a specific wine that is NOT in the user's journal database list above (e.g. "Tell me about Ridge Monte Bello 2019", "What is the blend of Château Latour 2015?").

Classify accurately.
"""

        response = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RoutingDecision,
                temperature=0.1
            )
        )
        return RoutingDecision.model_validate_json(response.text)

    # -----------------------------------------------------------------------
    # 2. Path A: Journal Query Execution
    # -----------------------------------------------------------------------

    def validate_sql_query(self, sql: str) -> Tuple[bool, Optional[str], str]:
        """
        Validates that SQL is a single, read-only SELECT query targeting only allowed tables.
        Returns: (is_valid, error_message, sanitized_sql)
        """
        clean_sql = sql.strip().rstrip(";")

        # Check for multiple statements
        if ";" in clean_sql:
            return False, "Query rejected: Multiple statements are not permitted.", clean_sql

        # Check for forbidden mutation keywords
        for pattern in FORBIDDEN_SQL_PATTERNS:
            match = re.search(pattern, clean_sql, re.IGNORECASE)
            if match:
                keyword = match.group(0).upper()
                return (
                    False,
                    f"Query rejected for security: Non-read-only operation detected ('{keyword}'). All queries in Ask & Research are strictly read-only SELECT statements.",
                    clean_sql
                )

        # Must start with SELECT or WITH
        if not re.match(r"^(SELECT|WITH)\b", clean_sql, re.IGNORECASE):
            return False, "Query rejected: Only SELECT queries are permitted.", clean_sql

        # Enforce LIMIT 50 if missing
        if not re.search(r"\bLIMIT\s+\d+", clean_sql, re.IGNORECASE):
            clean_sql += " LIMIT 50"

        return True, None, clean_sql

    def generate_and_execute_sql(self, user_query: str) -> Dict[str, Any]:
        """Generates SQL via Gemini, validates, executes read-only, and formats response."""
        # Check for hostile / mutation intent in the prompt directly
        hostile_match = re.search(r"\b(delete|drop|remove|truncate|clear|destroy|purge|wipe|insert|update|alter)\b", user_query, re.IGNORECASE)
        if hostile_match:
            kw = hostile_match.group(0).lower()
            refusal_msg = f"Operation refused: '{kw}' commands are not permitted. All queries in Ask & Research are strictly read-only. Journal data can only be modified through reviewed proposals."
            return {
                "path": "A",
                "path_label": "Path A: Journal Database Query",
                "success": False,
                "error": refusal_msg,
                "sql": f"-- REFUSED: {user_query}",
                "validation_failed": True,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "answer": refusal_msg
            }

        schema_desc = self.get_runtime_schema_description()

        prompt = f"""
You are an expert SQLite sommelier database engineer.
Generate a single read-only SQLite SELECT query to answer the user's question about their personal wine journal.

DATABASE SCHEMA:
{schema_desc}

KEY SCHEMA RELATIONSHIPS:
- `wines` links to `producers` via `wines.producer_id = producers.id`.
- `vintages` links to `wines` via `vintages.wine_id = wines.id`.
- `wine_blends` links to `vintages` via `wine_blends.vintage_id = vintages.id` and to `grape_varietals` via `wine_blends.varietal_id = grape_varietals.id`.
- `journal_entries` links to `vintages` via `journal_entries.vintage_id = vintages.id`.
- `terroirs` links to `wines` via `terroirs.wine_id = wines.id`.
- `vinifications` links to `wines` via `vinifications.wine_id = wines.id`.

IMPORTANT QUERY RULES:
1. Generate ONLY a single SQLite SELECT statement.
2. Case-insensitive matching: use `LIKE '%...%'` or `LOWER(column) = LOWER('...')`.
3. If the user asks for wine composition (e.g., at least 60% Cabernet Sauvignon), join `wine_blends` and `grape_varietals` and filter on `wine_blends.percentage`.
4. If the user asks about wines they have had or tasted, join `wines`, `vintages`, `producers`, and optionally `journal_entries`.
5. Select meaningful display columns: producer name, wine name, vintage year, appellation/region, and any queried metrics.
6. DO NOT use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, or ATTACH.
7. Return clean JSON matching the schema.

USER QUESTION:
"{user_query}"
"""

        gen_resp = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeneratedSQLQuery,
                temperature=0.1
            )
        )
        gen_data = GeneratedSQLQuery.model_validate_json(gen_resp.text)
        raw_sql = gen_data.sql

        # Validate SQL
        is_valid, err_msg, validated_sql = self.validate_sql_query(raw_sql)
        if not is_valid:
            return {
                "path": "A",
                "path_label": "Path A: Journal Database Query",
                "success": False,
                "error": err_msg,
                "sql": raw_sql,
                "validation_failed": True,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "answer": err_msg
            }

        # Execute read-only against SQLite
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = ON;")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(validated_sql)
            raw_rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [dict(r) for r in raw_rows]
            conn.close()
        except Exception as db_err:
            return {
                "path": "A",
                "path_label": "Path A: Journal Database Query",
                "success": False,
                "error": f"SQL execution error: {str(db_err)}",
                "sql": validated_sql,
                "validation_failed": False,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "answer": f"Unable to execute SQL query on your journal: {str(db_err)}"
            }

        # If zero rows returned:
        if len(rows) == 0:
            return {
                "path": "A",
                "path_label": "Path A: Journal Database Query",
                "success": True,
                "sql": validated_sql,
                "explanation": gen_data.explanation,
                "columns": columns,
                "rows": [],
                "row_count": 0,
                "is_empty": True,
                "answer": "Nothing in your journal matches this query.",
                "offer_path_b": True
            }

        # Generate grounded natural-language answer from the returned rows only
        answer_prompt = f"""
You are an expert AI Sommelier.
Answer the user's question directly and concisely using ONLY the structured data rows returned from their personal wine journal.

USER QUESTION:
"{user_query}"

SQL QUERY EXECUTED:
{validated_sql}

RETURNED DATA ROWS:
{json.dumps(rows, indent=2, default=str)}

INSTRUCTIONS:
- Ground your answer completely in the returned rows above.
- Mention specific estate names, vintages, appellations, and values from the rows.
- Do NOT fabricate or mention wines not present in the returned data.
"""
        ans_resp = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=answer_prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        answer_text = ans_resp.text.strip()

        return {
            "path": "A",
            "path_label": "Path A: Journal Database Query",
            "success": True,
            "sql": validated_sql,
            "explanation": gen_data.explanation,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "is_empty": False,
            "answer": answer_text
        }

    # -----------------------------------------------------------------------
    # 3. Path B: Deep Research Query Execution
    # -----------------------------------------------------------------------

    async def execute_research_query(
        self,
        user_query: str,
        wine_in_journal: bool = False,
        detected_wine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes search-grounded deep research via Gemini with Google Search tool grounding.
        Handles both general wine education queries and specific unrecorded wines.
        """
        if detected_wine:
            if hasattr(detected_wine, "model_dump"):
                dw_dict = detected_wine.model_dump()
            elif isinstance(detected_wine, dict):
                dw_dict = detected_wine
            else:
                dw_dict = {}
        else:
            dw_dict = {}

        is_specific_wine = bool(dw_dict and (dw_dict.get("wine_name") or dw_dict.get("producer")))
        sources: List[str] = []
        dossier_markdown = ""

        # Notice header if wine wasn't in journal
        not_in_db_notice = ""
        if is_specific_wine and not wine_in_journal:
            not_in_db_notice = "*(Note: This wine is not currently in your journal — here's what research found)*\n\n"

        if is_specific_wine:
            # Use full multi-domain research agent for the specific bottle
            producer = dw_dict.get("producer") or dw_dict.get("wine_name") or "Unknown"
            wine_name = dw_dict.get("wine_name") or dw_dict.get("producer") or "Grand Vin"
            vintage = dw_dict.get("vintage")

            report = await self.research_agent.conduct_deep_research(
                producer=producer,
                wine_name=wine_name,
                vintage=vintage
            )
            dossier_markdown = report.full_compiled_markdown
            sources = report.all_sources

            # Generate concise summary answer from the compiled dossier
            summary_prompt = f"""
Summarize the key viticultural and oenological facts for {producer} {wine_name} ({vintage or 'NV'}) based on this research dossier.
Highlight producer background, terroir/soils, vinification/aging, grape blend, and classical drinking window.

DOSSIER:
{dossier_markdown[:8000]}
"""
            sum_resp = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=summary_prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            answer_text = not_in_db_notice + sum_resp.text.strip()

            return {
                "path": "B",
                "path_label": "Path B: Deep Research Query",
                "success": True,
                "is_general_education": False,
                "is_specific_wine": True,
                "wine_in_journal": wine_in_journal,
                "answer": answer_text,
                "sources": sources,
                "can_save_to_journal": True,
                "wine_candidate": {
                    "producer": producer,
                    "wine_name": wine_name,
                    "vintage": vintage,
                    "dossier_markdown": dossier_markdown,
                    "sources": sources
                }
            }

        else:
            # General wine education query (e.g. Margaux vs Pauillac, Malolactic conversion, etc.)
            educational_prompt = f"""
You are an expert viticulture educator and master sommelier.
Perform web searches to reference authoritative viticultural literature, terroir analyses, and technical sources to answer the user's educational question thoroughly and accurately.

USER QUESTION:
"{user_query}"

INSTRUCTIONS:
- Ground your answer in search results.
- Explain historical, geological, terroir, stylistic, and technical winemaking reasons.
- Provide concrete examples, soil types, grape varieties, and structural sensory differences.
- Avoid generic filler.
"""
            gen_resp = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=educational_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1
                )
            )

            # Extract search grounding sources
            if gen_resp.candidates and gen_resp.candidates[0].grounding_metadata:
                metadata = gen_resp.candidates[0].grounding_metadata
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web and chunk.web.uri:
                            sources.append(chunk.web.uri)
            sources = sorted(list(set(sources)))

            answer_text = gen_resp.text.strip()

            return {
                "path": "B",
                "path_label": "Path B: Wine Research & Education",
                "success": True,
                "is_general_education": True,
                "is_specific_wine": False,
                "wine_in_journal": False,
                "answer": answer_text,
                "sources": sources,
                "can_save_to_journal": False
            }

    # -----------------------------------------------------------------------
    # 4. Path C: Save What You Learned (Create Proposal from Research)
    # -----------------------------------------------------------------------

    def create_proposal_from_research(
        self,
        producer_name: str,
        wine_name: str,
        vintage_year: Optional[int],
        dossier_markdown: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """
        Executes Path C: Converts Path B research into a pending DossierProposal
        linked to draft records in the database with 'ask_and_research' provenance.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Match or Create Producer
        cursor.execute("SELECT id FROM producers WHERE LOWER(name) = LOWER(?) LIMIT 1;", (producer_name.strip(),))
        prod_row = cursor.fetchone()
        if prod_row:
            producer_id = prod_row[0]
        else:
            cursor.execute(
                "INSERT INTO producers (name, country, region) VALUES (?, ?, ?);",
                (producer_name.strip(), "Unknown", "Unknown")
            )
            producer_id = cursor.lastrowid

        # 2. Match or Create Wine
        cursor.execute(
            "SELECT id FROM wines WHERE producer_id = ? AND LOWER(name) = LOWER(?) LIMIT 1;",
            (producer_id, wine_name.strip())
        )
        wine_row = cursor.fetchone()
        if wine_row:
            wine_id = wine_row[0]
        else:
            cursor.execute(
                "INSERT INTO wines (producer_id, name, wine_type) VALUES (?, ?, ?);",
                (producer_id, wine_name.strip(), "RED")
            )
            wine_id = cursor.lastrowid

        # 3. Match or Create Vintage
        if vintage_year:
            cursor.execute("SELECT id FROM vintages WHERE wine_id = ? AND year = ? LIMIT 1;", (wine_id, vintage_year))
        else:
            cursor.execute("SELECT id FROM vintages WHERE wine_id = ? AND year IS NULL LIMIT 1;", (wine_id,))
        vint_row = cursor.fetchone()
        if vint_row:
            vintage_id = vint_row[0]
        else:
            cursor.execute("INSERT INTO vintages (wine_id, year) VALUES (?, ?);", (wine_id, vintage_year))
            vintage_id = cursor.lastrowid

        # 4. Create or Update ResearchCorpus with provenance="ask_and_research"
        cursor.execute("SELECT id FROM research_corpora WHERE vintage_id = ? LIMIT 1;", (vintage_id,))
        corp_row = cursor.fetchone()
        sources_json_str = json.dumps(sources)
        if corp_row:
            cursor.execute("""
                UPDATE research_corpora 
                SET full_research_markdown = ?, sources_json = ?, provenance = 'ask_and_research'
                WHERE vintage_id = ?;
            """, (dossier_markdown, sources_json_str, vintage_id))
        else:
            cursor.execute("""
                INSERT INTO research_corpora (vintage_id, full_research_markdown, sources_json, provenance)
                VALUES (?, ?, ?, 'ask_and_research');
            """, (vintage_id, dossier_markdown, sources_json_str))

        conn.commit()
        conn.close()

        # 5. Extract structured proposal via DossierExtractor
        extractor = DossierExtractor(api_key=self.api_key)
        extraction = extractor.extract(
            corpus_markdown=dossier_markdown,
            producer_name=producer_name,
            wine_name=wine_name,
            vintage_year=vintage_year
        )
        proposal_dict = extraction.model_dump()

        # 6. Save DossierProposal in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Supersede existing pending proposals
        cursor.execute(
            "UPDATE dossier_proposals SET status = 'rejected', resolved_at = datetime('now') WHERE vintage_id = ? AND status = 'pending';",
            (vintage_id,)
        )
        cursor.execute(
            "INSERT INTO dossier_proposals (vintage_id, proposal_json, status) VALUES (?, ?, 'pending');",
            (vintage_id, json.dumps(proposal_dict))
        )
        proposal_id = cursor.lastrowid
        conn.commit()

        # Build current data for proposal comparison UI
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM producers WHERE id = ?", (producer_id,))
        p_row = cur.fetchone()
        cur.execute("SELECT * FROM wines WHERE id = ?", (wine_id,))
        w_row = cur.fetchone()
        cur.execute("SELECT * FROM terroirs WHERE wine_id = ?", (wine_id,))
        t_row = cur.fetchone()
        cur.execute("SELECT * FROM vinifications WHERE wine_id = ?", (wine_id,))
        v_row = cur.fetchone()
        cur.execute("SELECT * FROM vintages WHERE id = ?", (vintage_id,))
        vint_row = cur.fetchone()
        cur.execute("""
            SELECT wb.percentage, gv.name as varietal_name 
            FROM wine_blends wb 
            JOIN grape_varietals gv ON wb.varietal_id = gv.id 
            WHERE wb.vintage_id = ?
        """, (vintage_id,))
        blends_rows = cur.fetchall()

        current_data = {
            "producer": {
                "canonical_name": p_row["canonical_name"] if p_row else None,
                "country": p_row["country"] if p_row else None,
                "region": p_row["region"] if p_row else None,
                "subregion": p_row["subregion"] if p_row else None,
                "farming_practices": p_row["farming_practices"] if p_row else None,
                "certifications": json.loads(p_row["certifications"]) if (p_row and p_row["certifications"]) else [],
                "winemaker_philosophy": p_row["winemaker_philosophy"] if p_row else None,
                "history_summary": p_row["history_md"] if p_row else None,
            },
            "wine": {
                "classification": w_row["classification"] if w_row else None,
                "appellation": w_row["appellation"] if w_row else None,
                "wine_type": w_row["wine_type"] if w_row else "red",
            },
            "terroir": {
                "soil_types": json.loads(t_row["soil_types"]) if (t_row and t_row["soil_types"]) else [],
                "elevation_meters": t_row["elevation_meters"] if t_row else None,
                "aspect": t_row["aspect"] if t_row else None,
                "single_vineyard_name": t_row["single_vineyard_name"] if t_row else None,
            },
            "vinification": {
                "fermentation_vessel": v_row["fermentation_vessel"] if v_row else None,
                "yeast_type": v_row["yeast_type"] if v_row else None,
                "oak_type": v_row["oak_type"] if v_row else None,
                "new_oak_percent": v_row["new_oak_percent"] if v_row else None,
                "aging_months": v_row["aging_months"] if v_row else None,
                "malolactic_fermentation": v_row["malolactic_fermentation"] if v_row else None,
                "filtration_notes": v_row["filtration_notes"] if v_row else None,
            },
            "vintage": {
                "abv": vint_row["abv"] if vint_row else None,
                "drinking_window_start": vint_row["drinking_window_start"] if vint_row else None,
                "drinking_window_end": vint_row["drinking_window_end"] if vint_row else None,
                "weather_summary": vint_row["weather_summary"] if vint_row else None,
            },
            "blend": [
                {"varietal_name": b["varietal_name"], "percentage": b["percentage"]}
                for b in blends_rows
            ],
            "predicted_profile": json.loads(vint_row["predicted_profile_json"]) if (vint_row and vint_row["predicted_profile_json"]) else {}
        }
        conn.close()

        return {
            "proposal_id": proposal_id,
            "vintage_id": vintage_id,
            "wine_id": wine_id,
            "producer_id": producer_id,
            "status": "pending",
            "proposed": proposal_dict,
            "current": current_data
        }
