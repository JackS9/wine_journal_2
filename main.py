"""
FastAPI Application Backend for AI Wine Journal
Handles label image uploads, agent streaming research (SSE), database persistence,
structured dossier extraction & proposals, cellar management, and conversational Q&A.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select, or_, and_, func, desc
from sqlalchemy.orm import sessionmaker, joinedload

# Import models
from models import (
    Base, Producer, Wine, Vintage, VivinoMetadata, ResearchCorpus,
    JournalEntry, TastingProfile, Terroir, Vinification, WineBlend,
    GrapeVarietal, DossierProposal, WineType, FarmingPractice
)
from vivino_pipeline import WineLabelIngestionPipeline
from research_agent import WineResearchAgent
from dossier_extractor import DossierExtractor, extract_dossier_from_corpus

# Initialize FastAPI App
app = FastAPI(title="AI Agentic Wine Journal", version="2.0")

# Setup SQLite Database Engine
DB_PATH = "wine_journal.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables exist
Base.metadata.create_all(bind=engine)

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pipeline instances
pipeline = WineLabelIngestionPipeline()
research_agent = WineResearchAgent()
dossier_extractor = DossierExtractor()

# Static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def compute_drink_status(start: Optional[int], end: Optional[int], current_year: Optional[int] = None) -> str:
    """Computes ready-to-drink status based on drinking window and current year."""
    if current_year is None:
        current_year = datetime.now().year
    if start is None or end is None:
        return "unknown"
    if current_year < start:
        return "too_young"
    if current_year > end:
        return "past_peak"
    return "in_window"


def get_latest_label_photo(db, wine_id: int) -> Optional[str]:
    """Finds the most recent journal entry label photo for a wine."""
    stmt = (
        select(JournalEntry.label_photo_path)
        .join(Vintage, JournalEntry.vintage_id == Vintage.id)
        .where(Vintage.wine_id == wine_id, JournalEntry.label_photo_path.isnot(None))
        .order_by(desc(JournalEntry.drank_at))
        .limit(1)
    )
    return db.scalar(stmt)


# ---------------------------------------------------------------------------
# Request/Response Pydantic Schemas
# ---------------------------------------------------------------------------

class TastingEntryCreate(BaseModel):
    vintage_id: int
    drank_at: Optional[str] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    occasion: Optional[str] = None
    companions: Optional[str] = None
    price_paid: Optional[float] = None
    personal_rating: Optional[float] = None
    user_notes: Optional[str] = None
    label_photo_path: Optional[str] = None
    # WSET SAT Attributes
    color_hue: Optional[str] = None
    nose_intensity: Optional[str] = None
    palate_sweetness: Optional[str] = None
    palate_acidity: Optional[str] = None
    palate_tannin: Optional[str] = None
    palate_body: Optional[str] = None
    palate_finish: Optional[str] = None
    overall_quality: Optional[str] = None


class ChatQuery(BaseModel):
    vintage_id: int
    user_message: str


class AcceptProposalPayload(BaseModel):
    fields_accepted: Optional[Dict[str, Any]] = None
    append_unmapped_to_notes: bool = False
    chosen_conflicts: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# API Endpoints: Label Scan & Autonomous Research
# ---------------------------------------------------------------------------

@app.post("/api/journal/scan")
async def scan_label_and_initiate_research(file: UploadFile = File(...)):
    """
    Accepts label photo, runs Vision OCR + Vivino scraping, matches existing producer
    or creates DB entities, and returns vintage_id and reuse information.
    """
    # 1. Save uploaded image to disk
    file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = UPLOAD_DIR / f"label_{timestamp}.{file_ext}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Run Vision OCR & Vivino pipeline
    label_info, vivino_data = pipeline.process_image(str(file_path))

    # 3. Store in Database with Producer Reuse
    db = SessionLocal()
    try:
        # Match existing producer by name or canonical_name (case-insensitive)
        cleaned_producer_name = label_info.producer.strip()
        producer = db.scalar(
            select(Producer).where(
                or_(
                    func.lower(Producer.name) == func.lower(cleaned_producer_name),
                    func.lower(Producer.canonical_name) == func.lower(cleaned_producer_name)
                )
            )
        )

        producer_reused = False
        if producer:
            producer_reused = True
        else:
            producer = Producer(
                name=cleaned_producer_name,
                country=label_info.country or "Unknown",
                region=label_info.region or "Unknown"
            )
            db.add(producer)
            db.flush()

        # Producer already researched check
        producer_already_researched = bool(
            (producer.history_md and len(producer.history_md.strip()) > 0) or
            (producer.winemaker_philosophy and len(producer.winemaker_philosophy.strip()) > 0)
        )

        # Wine Match or Creation
        cleaned_wine_name = label_info.wine_name.strip()
        wine = db.scalar(
            select(Wine).where(
                Wine.producer_id == producer.id,
                func.lower(Wine.name) == func.lower(cleaned_wine_name)
            )
        )
        if not wine:
            wine = Wine(
                producer_id=producer.id,
                name=cleaned_wine_name,
                wine_type=label_info.wine_type.lower(),
                appellation=label_info.appellation
            )
            db.add(wine)
            db.flush()

        # Vintage Match or Creation
        vintage = db.scalar(
            select(Vintage).where(
                Vintage.wine_id == wine.id,
                Vintage.year == label_info.vintage
            )
        )
        if not vintage:
            vintage = Vintage(wine_id=wine.id, year=label_info.vintage)
            db.add(vintage)
            db.flush()

        # Vivino Metadata if new
        if vivino_data and (vivino_data.rating_average or vivino_data.structure.boldness):
            existing_vivino = db.scalar(select(VivinoMetadata).where(VivinoMetadata.vintage_id == vintage.id))
            if not existing_vivino:
                vivino_record = VivinoMetadata(
                    vintage_id=vintage.id,
                    vivino_id=vivino_data.vivino_id or "N/A",
                    page_url=vivino_data.page_url,
                    rating_average=vivino_data.rating_average,
                    ratings_count=vivino_data.ratings_count,
                    price_usd=vivino_data.price_usd,
                    boldness=vivino_data.structure.boldness,
                    tannin=vivino_data.structure.tannin,
                    sweetness=vivino_data.structure.sweetness,
                    acidity=vivino_data.structure.acidity,
                    raw_js_data=vivino_data.raw_next_data
                )
                db.add(vivino_record)

        db.commit()
        vintage_id = vintage.id
        wine_id = wine.id
        producer_name = producer.name

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error during scan: {str(e)}")
    finally:
        db.close()

    return {
        "vintage_id": vintage_id,
        "wine_id": wine_id,
        "producer_id": producer.id,
        "producer_name": producer_name,
        "producer_reused": producer_reused,
        "producer_already_researched": producer_already_researched,
        "image_url": f"/uploads/{file_path.name}",
        "ocr_extracted": label_info.model_dump(),
        "vivino_data": vivino_data.model_dump(exclude={"raw_next_data"}) if vivino_data else {}
    }


@app.get("/api/journal/research-stream/{vintage_id}")
async def research_agent_stream(vintage_id: int, skip_producer: bool = Query(False)):
    """
    Streams agent research progress in real-time via Server-Sent Events (SSE).
    Populates sources_json from agent grounding sources.
    """
    db = SessionLocal()
    vintage = db.scalar(
        select(Vintage)
        .options(joinedload(Vintage.wine).joinedload(Wine.producer))
        .where(Vintage.id == vintage_id)
    )
    if not vintage:
        db.close()
        raise HTTPException(status_code=404, detail="Vintage record not found")

    producer_name = vintage.wine.producer.name
    wine_name = vintage.wine.name
    year = vintage.year
    region = vintage.wine.producer.region
    appellation = vintage.wine.appellation
    db.close()

    async def event_generator():
        skip_msg = " (skipping producer domain — existing history retained)" if skip_producer else ""
        yield f"data: {json.dumps({'status': 'started', 'message': f'Launching AI research agent for {producer_name} {wine_name}{skip_msg}...'})}\n\n"
        await asyncio.sleep(0.5)

        yield f"data: {json.dumps({'status': 'agent_running', 'message': 'Searching winery history, terroir profiles, vintage weather, and tech sheets...'})}\n\n"

        report = await research_agent.conduct_deep_research(
            producer=producer_name,
            wine_name=wine_name,
            vintage=year,
            region=region,
            appellation=appellation,
            skip_producer=skip_producer
        )

        # Save or update research report in database with sources_json
        db_session = SessionLocal()
        try:
            corpus = db_session.scalar(select(ResearchCorpus).where(ResearchCorpus.vintage_id == vintage_id))
            if not corpus:
                corpus = ResearchCorpus(vintage_id=vintage_id)
                db_session.add(corpus)

            corpus.winery_history_md = report.producer_philosophy.summary_md
            corpus.terroir_deep_dive_md = report.terroir_and_climate.summary_md
            corpus.vintage_weather_report_md = report.terroir_and_climate.summary_md
            corpus.winemaking_tech_sheet_md = report.vinification_and_aging.summary_md
            corpus.full_research_markdown = report.full_compiled_markdown
            corpus.sources_json = report.all_sources

            db_session.commit()
        except Exception as e:
            db_session.rollback()
            print(f"Error saving corpus: {e}")
        finally:
            db_session.close()

        yield f"data: {json.dumps({'status': 'completed', 'markdown': report.full_compiled_markdown, 'sources': report.all_sources})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# API Endpoints: Extraction & Proposal Review
# ---------------------------------------------------------------------------

@app.post("/api/vintages/{vintage_id}/extract")
async def extract_dossier_endpoint(vintage_id: int):
    """
    Extracts structured dossier facts from the saved research corpus.
    Supersedes any existing pending proposals for this vintage as rejected.
    Returns the new pending proposal with side-by-side proposed vs current values.
    """
    db = SessionLocal()
    try:
        vintage = db.scalar(
            select(Vintage)
            .options(
                joinedload(Vintage.wine).joinedload(Wine.producer),
                joinedload(Vintage.wine).joinedload(Wine.terroir),
                joinedload(Vintage.wine).joinedload(Wine.vinification),
                joinedload(Vintage.research_corpus),
                joinedload(Vintage.blends).joinedload(WineBlend.varietal)
            )
            .where(Vintage.id == vintage_id)
        )
        if not vintage:
            raise HTTPException(status_code=404, detail="Vintage not found")

        if not vintage.research_corpus:
            raise HTTPException(status_code=400, detail="No research corpus exists for this vintage yet. Run research first.")

        # Supersede any existing pending proposals
        existing_pendings = db.scalars(
            select(DossierProposal)
            .where(DossierProposal.vintage_id == vintage_id, DossierProposal.status == "pending")
        ).all()
        for p in existing_pendings:
            p.status = "rejected"
            p.resolved_at = datetime.utcnow()

        wine = vintage.wine
        producer = wine.producer
        corpus_md = vintage.research_corpus.full_research_markdown

        # Call Dossier Extractor
        extraction = extract_dossier_from_corpus(
            corpus_markdown=corpus_md,
            producer_name=producer.name,
            wine_name=wine.name,
            vintage_year=vintage.year
        )

        proposal_dict = extraction.model_dump()

        # Save new DossierProposal
        new_proposal = DossierProposal(
            vintage_id=vintage_id,
            proposal_json=proposal_dict,
            status="pending"
        )
        db.add(new_proposal)
        db.commit()
        db.refresh(new_proposal)

        # Build Current vs Proposed comparison
        current_data = {
            "producer": {
                "canonical_name": producer.canonical_name,
                "country": producer.country,
                "region": producer.region,
                "subregion": producer.subregion,
                "farming_practices": producer.farming_practices.value if producer.farming_practices else None,
                "certifications": producer.certifications or [],
                "winemaker_philosophy": producer.winemaker_philosophy,
                "history_summary": producer.history_md
            },
            "wine": {
                "classification": wine.classification,
                "appellation": wine.appellation,
                "wine_type": wine.wine_type.value if hasattr(wine.wine_type, "value") else str(wine.wine_type)
            },
            "terroir": {
                "soil_types": wine.terroir.soil_types if wine.terroir else [],
                "elevation_meters": wine.terroir.elevation_meters if wine.terroir else None,
                "aspect": wine.terroir.aspect if wine.terroir else None,
                "single_vineyard_name": wine.terroir.single_vineyard_name if wine.terroir else None
            },
            "vinification": {
                "fermentation_vessel": wine.vinification.fermentation_vessel if wine.vinification else None,
                "yeast_type": wine.vinification.yeast_type if wine.vinification else None,
                "oak_type": wine.vinification.oak_type if wine.vinification else None,
                "new_oak_percent": wine.vinification.new_oak_percent if wine.vinification else None,
                "aging_months": wine.vinification.aging_months if wine.vinification else None,
                "malolactic_fermentation": wine.vinification.malolactic_fermentation if wine.vinification else None,
                "filtration_notes": wine.vinification.filtration_notes if wine.vinification else None
            },
            "vintage": {
                "abv": vintage.abv,
                "drinking_window_start": vintage.drinking_window_start,
                "drinking_window_end": vintage.drinking_window_end,
                "weather_summary": vintage.weather_summary
            },
            "blend": [
                {"varietal_name": b.varietal.name, "percentage": b.percentage}
                for b in vintage.blends
            ],
            "predicted_profile": vintage.predicted_profile_json or {}
        }

        return {
            "proposal_id": new_proposal.id,
            "vintage_id": vintage_id,
            "status": new_proposal.status,
            "created_at": new_proposal.created_at.isoformat(),
            "proposed": proposal_dict,
            "current": current_data
        }

    finally:
        db.close()


@app.post("/api/proposals/{proposal_id}/accept")
def accept_proposal_endpoint(proposal_id: int, payload: AcceptProposalPayload = Body(...)):
    """
    Applies accepted fields from the proposal to Producer, Wine, Terroir, Vinification,
    Vintage, WineBlend, and sets status to 'accepted'.
    """
    db = SessionLocal()
    try:
        proposal = db.scalar(
            select(DossierProposal)
            .options(
                joinedload(DossierProposal.vintage)
                .joinedload(Vintage.wine)
                .joinedload(Wine.producer),
                joinedload(DossierProposal.vintage).joinedload(Vintage.wine).joinedload(Wine.terroir),
                joinedload(DossierProposal.vintage).joinedload(Vintage.wine).joinedload(Wine.vinification)
            )
            .where(DossierProposal.id == proposal_id)
        )
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")

        if proposal.status == "accepted":
            return {"status": "already_accepted", "proposal_id": proposal_id}

        vintage = proposal.vintage
        wine = vintage.wine
        producer = wine.producer
        data = payload.fields_accepted or proposal.proposal_json

        # 1. Update Producer
        prod_data = data.get("producer", {})
        if prod_data.get("canonical_name"):
            producer.canonical_name = prod_data["canonical_name"]
        if prod_data.get("country") and (not producer.country or producer.country == "Unknown"):
            producer.country = prod_data["country"]
        if prod_data.get("region") and (not producer.region or producer.region == "Unknown"):
            producer.region = prod_data["region"]
        if prod_data.get("subregion"):
            producer.subregion = prod_data["subregion"]
        if prod_data.get("farming_practices"):
            try:
                producer.farming_practices = FarmingPractice(prod_data["farming_practices"].lower())
            except Exception:
                pass
        if prod_data.get("certifications"):
            producer.certifications = prod_data["certifications"]
        if prod_data.get("winemaker_philosophy"):
            producer.winemaker_philosophy = prod_data["winemaker_philosophy"]
        if prod_data.get("history_summary"):
            producer.history_md = prod_data["history_summary"]

        # 2. Update Wine
        wine_data = data.get("wine", {})
        if wine_data.get("classification"):
            wine.classification = wine_data["classification"]
        if wine_data.get("appellation"):
            wine.appellation = wine_data["appellation"]
        if wine_data.get("wine_type"):
            try:
                wine.wine_type = WineType(wine_data["wine_type"].lower())
            except Exception:
                pass

        # 3. Update or Create Terroir (on Wine)
        terroir_data = data.get("terroir", {})
        if terroir_data:
            if not wine.terroir:
                wine.terroir = Terroir(wine_id=wine.id)
                db.add(wine.terroir)
            if terroir_data.get("soil_types"):
                wine.terroir.soil_types = terroir_data["soil_types"]
            if terroir_data.get("elevation_meters") is not None:
                wine.terroir.elevation_meters = float(terroir_data["elevation_meters"])
            if terroir_data.get("aspect"):
                wine.terroir.aspect = terroir_data["aspect"]
            if terroir_data.get("single_vineyard_name"):
                wine.terroir.single_vineyard_name = terroir_data["single_vineyard_name"]

        # 4. Update or Create Vinification (on Wine)
        vinif_data = data.get("vinification", {})
        if vinif_data:
            if not wine.vinification:
                wine.vinification = Vinification(wine_id=wine.id)
                db.add(wine.vinification)
            if vinif_data.get("fermentation_vessel"):
                wine.vinification.fermentation_vessel = vinif_data["fermentation_vessel"]
            if vinif_data.get("yeast_type"):
                wine.vinification.yeast_type = vinif_data["yeast_type"]
            if vinif_data.get("oak_type"):
                wine.vinification.oak_type = vinif_data["oak_type"]
            if vinif_data.get("new_oak_percent") is not None:
                wine.vinification.new_oak_percent = float(vinif_data["new_oak_percent"])
            if vinif_data.get("aging_months") is not None:
                wine.vinification.aging_months = int(vinif_data["aging_months"])
            if vinif_data.get("malolactic_fermentation") is not None:
                wine.vinification.malolactic_fermentation = bool(vinif_data["malolactic_fermentation"])
            if vinif_data.get("filtration_notes"):
                wine.vinification.filtration_notes = vinif_data["filtration_notes"]

        # 5. Update Vintage
        vint_data = data.get("vintage", {})
        if vint_data.get("abv") is not None:
            vintage.abv = float(vint_data["abv"])
        if vint_data.get("drinking_window_start") is not None:
            vintage.drinking_window_start = int(vint_data["drinking_window_start"])
        if vint_data.get("drinking_window_end") is not None:
            vintage.drinking_window_end = int(vint_data["drinking_window_end"])
        if vint_data.get("weather_summary"):
            vintage.weather_summary = vint_data["weather_summary"]

        # Predicted Profile
        pred_profile = data.get("predicted_profile")
        if pred_profile:
            vintage.predicted_profile_json = pred_profile

        # 6. Replace WineBlend for this Vintage
        blend_list = data.get("blend", [])
        if blend_list:
            # Delete existing blends
            db.query(WineBlend).filter(WineBlend.vintage_id == vintage.id).delete()
            db.flush()
            for b in blend_list:
                var_name = b.get("varietal_name", "").strip()
                if not var_name:
                    continue
                varietal = db.scalar(
                    select(GrapeVarietal).where(func.lower(GrapeVarietal.name) == func.lower(var_name))
                )
                if not varietal:
                    varietal = GrapeVarietal(name=var_name)
                    db.add(varietal)
                    db.flush()
                wb = WineBlend(
                    vintage_id=vintage.id,
                    varietal_id=varietal.id,
                    percentage=float(b["percentage"]) if b.get("percentage") is not None else None
                )
                db.add(wb)

        # 7. Unmapped facts append option
        if payload.append_unmapped_to_notes:
            unmapped = proposal.proposal_json.get("unmapped_facts", [])
            if unmapped:
                unmapped_text = "\n\n### Salient Tech Notes:\n" + "\n".join(f"- {fact}" for fact in unmapped)
                if vintage.weather_summary:
                    vintage.weather_summary += unmapped_text
                else:
                    vintage.weather_summary = unmapped_text

        # Mark proposal accepted
        proposal.status = "accepted"
        proposal.resolved_at = datetime.utcnow()

        db.commit()
        return {
            "status": "accepted",
            "proposal_id": proposal_id,
            "vintage_id": vintage.id,
            "wine_id": wine.id
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to accept proposal: {str(e)}")
    finally:
        db.close()


@app.post("/api/proposals/{proposal_id}/reject")
def reject_proposal_endpoint(proposal_id: int):
    """Marks proposal as rejected without modifying data."""
    db = SessionLocal()
    try:
        proposal = db.get(DossierProposal, proposal_id)
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
        proposal.status = "rejected"
        proposal.resolved_at = datetime.utcnow()
        db.commit()
        return {"status": "rejected", "proposal_id": proposal_id}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# API Endpoints: Cellar Browse, Search, Wine Detail, & Journal List
# ---------------------------------------------------------------------------

@app.get("/api/wines")
def list_wines(
    q: Optional[str] = Query(None),
    wine_type: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    drink_status: Optional[str] = Query(None)
):
    """
    Lists wines for the Cellar view with search (q), filters, and drink_status badges.
    """
    db = SessionLocal()
    try:
        current_year = datetime.now().year
        stmt = (
            select(Wine)
            .options(
                joinedload(Wine.producer),
                joinedload(Wine.terroir),
                joinedload(Wine.vinification),
                joinedload(Wine.vintages).joinedload(Vintage.blends).joinedload(WineBlend.varietal),
                joinedload(Wine.vintages).joinedload(Vintage.research_corpus),
                joinedload(Wine.vintages).joinedload(Vintage.proposals)
            )
        )

        # Filters
        if wine_type:
            stmt = stmt.where(func.lower(Wine.wine_type) == func.lower(wine_type))
        if region:
            stmt = stmt.join(Producer).where(func.lower(Producer.region).contains(region.lower()))

        wines = db.scalars(stmt).unique().all()

        results = []
        for w in wines:
            # Check varietals for q search
            varietal_names = []
            vintage_summaries = []
            has_in_window = False

            for v in sorted(w.vintages, key=lambda x: (x.year or 0)):
                v_status = compute_drink_status(v.drinking_window_start, v.drinking_window_end, current_year)
                if v_status == "in_window":
                    has_in_window = True

                blend_items = []
                for b in v.blends:
                    if b.varietal:
                        varietal_names.append(b.varietal.name)
                        blend_items.append({"varietal": b.varietal.name, "percentage": b.percentage})

                vintage_summaries.append({
                    "id": v.id,
                    "year": v.year,
                    "abv": v.abv,
                    "drinking_window_start": v.drinking_window_start,
                    "drinking_window_end": v.drinking_window_end,
                    "drink_status": v_status,
                    "has_corpus": v.research_corpus is not None,
                    "has_proposal": len(v.proposals) > 0,
                    "weather_summary": v.weather_summary,
                    "blend": blend_items
                })

            # Text Search (q) across producer, wine name, appellation, region, varietals
            if q:
                search_term = q.strip().lower()
                matches = (
                    search_term in w.name.lower() or
                    search_term in (w.producer.name or "").lower() or
                    search_term in (w.producer.canonical_name or "").lower() or
                    search_term in (w.producer.region or "").lower() or
                    search_term in (w.appellation or "").lower() or
                    any(search_term in vn.lower() for vn in varietal_names)
                )
                if not matches:
                    continue

            # Drink status filter
            if drink_status:
                if drink_status == "in_window" and not has_in_window:
                    continue
                elif drink_status != "in_window":
                    matches_status = any(vs["drink_status"] == drink_status for vs in vintage_summaries)
                    if not matches_status:
                        continue

            latest_photo = get_latest_label_photo(db, w.id)

            results.append({
                "id": w.id,
                "name": w.name,
                "wine_type": w.wine_type.value if hasattr(w.wine_type, "value") else str(w.wine_type),
                "classification": w.classification,
                "appellation": w.appellation,
                "producer": {
                    "id": w.producer.id,
                    "name": w.producer.name,
                    "canonical_name": w.producer.canonical_name,
                    "country": w.producer.country,
                    "region": w.producer.region,
                    "subregion": w.producer.subregion
                },
                "has_terroir": w.terroir is not None,
                "has_vinification": w.vinification is not None,
                "vintages": vintage_summaries,
                "latest_label_photo": latest_photo
            })

        return results
    finally:
        db.close()


@app.get("/api/wines/{wine_id}")
def get_wine_detail(wine_id: int):
    """
    Returns full aggregate view for a wine: producer dossier, terroir, vinification,
    per-vintage panels (blends, windows, weather, predicted profiles), research corpora,
    proposals, and tasting journal entries.
    """
    db = SessionLocal()
    try:
        current_year = datetime.now().year
        wine = db.scalar(
            select(Wine)
            .options(
                joinedload(Wine.producer),
                joinedload(Wine.terroir),
                joinedload(Wine.vinification),
                joinedload(Wine.vintages).joinedload(Vintage.blends).joinedload(WineBlend.varietal),
                joinedload(Wine.vintages).joinedload(Vintage.research_corpus),
                joinedload(Wine.vintages).joinedload(Vintage.proposals),
                joinedload(Wine.vintages).joinedload(Vintage.journal_entries).joinedload(JournalEntry.tasting_profile),
                joinedload(Wine.vintages).joinedload(Vintage.vivino_data)
            )
            .where(Wine.id == wine_id)
        )
        if not wine:
            raise HTTPException(status_code=404, detail="Wine not found")

        producer = wine.producer
        terroir = wine.terroir
        vinification = wine.vinification

        vintages_data = []
        for v in sorted(wine.vintages, key=lambda x: (x.year or 0), reverse=True):
            status = compute_drink_status(v.drinking_window_start, v.drinking_window_end, current_year)
            
            blends_data = [
                {"varietal_name": b.varietal.name, "percentage": b.percentage}
                for b in v.blends if b.varietal
            ]

            corpus_data = None
            if v.research_corpus:
                corpus_data = {
                    "id": v.research_corpus.id,
                    "winery_history_md": v.research_corpus.winery_history_md,
                    "terroir_deep_dive_md": v.research_corpus.terroir_deep_dive_md,
                    "vintage_weather_report_md": v.research_corpus.vintage_weather_report_md,
                    "winemaking_tech_sheet_md": v.research_corpus.winemaking_tech_sheet_md,
                    "full_research_markdown": v.research_corpus.full_research_markdown,
                    "sources_json": v.research_corpus.sources_json or []
                }

            # Latest proposal if any
            latest_proposal = None
            if v.proposals:
                sorted_props = sorted(v.proposals, key=lambda p: p.id, reverse=True)
                p = sorted_props[0]
                latest_proposal = {
                    "id": p.id,
                    "status": p.status,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "resolved_at": p.resolved_at.isoformat() if p.resolved_at else None,
                    "proposal_json": p.proposal_json
                }

            # Journal entries
            entries_data = []
            for j in sorted(v.journal_entries, key=lambda x: x.drank_at, reverse=True):
                tp_data = None
                if j.tasting_profile:
                    tp = j.tasting_profile
                    tp_data = {
                        "color_hue": tp.color_hue,
                        "nose_intensity": tp.nose_intensity.value if tp.nose_intensity else None,
                        "palate_sweetness": tp.palate_sweetness,
                        "palate_acidity": tp.palate_acidity.value if tp.palate_acidity else None,
                        "palate_tannin": tp.palate_tannin.value if tp.palate_tannin else None,
                        "palate_body": tp.palate_body.value if tp.palate_body else None,
                        "palate_finish": tp.palate_finish.value if tp.palate_finish else None,
                        "overall_quality": tp.overall_quality.value if tp.overall_quality else None,
                        "aroma_descriptors": tp.aroma_descriptors or {}
                    }
                entries_data.append({
                    "id": j.id,
                    "drank_at": j.drank_at.isoformat() if j.drank_at else None,
                    "location_name": j.location_name,
                    "occasion": j.occasion,
                    "companions": j.companions,
                    "price_paid": j.price_paid,
                    "personal_rating": j.personal_rating,
                    "user_notes": j.user_notes,
                    "label_photo_path": j.label_photo_path,
                    "tasting_profile": tp_data
                })

            vintages_data.append({
                "id": v.id,
                "year": v.year,
                "abv": v.abv,
                "drinking_window_start": v.drinking_window_start,
                "drinking_window_end": v.drinking_window_end,
                "drink_status": status,
                "weather_summary": v.weather_summary,
                "predicted_profile_json": v.predicted_profile_json,
                "blends": blends_data,
                "research_corpus": corpus_data,
                "latest_proposal": latest_proposal,
                "journal_entries": entries_data
            })

        latest_label_photo = get_latest_label_photo(db, wine.id)

        return {
            "id": wine.id,
            "name": wine.name,
            "wine_type": wine.wine_type.value if hasattr(wine.wine_type, "value") else str(wine.wine_type),
            "classification": wine.classification,
            "appellation": wine.appellation,
            "latest_label_photo": latest_label_photo,
            "producer": {
                "id": producer.id,
                "name": producer.name,
                "canonical_name": producer.canonical_name,
                "country": producer.country,
                "region": producer.region,
                "subregion": producer.subregion,
                "farming_practices": producer.farming_practices.value if producer.farming_practices else None,
                "certifications": producer.certifications or [],
                "winemaker_philosophy": producer.winemaker_philosophy,
                "history_md": producer.history_md,
                "website_url": producer.website_url
            },
            "terroir": {
                "single_vineyard_name": terroir.single_vineyard_name if terroir else None,
                "soil_types": terroir.soil_types if terroir else [],
                "elevation_meters": terroir.elevation_meters if terroir else None,
                "aspect": terroir.aspect if terroir else None
            } if terroir else None,
            "vinification": {
                "fermentation_vessel": vinification.fermentation_vessel if vinification else None,
                "yeast_type": vinification.yeast_type if vinification else None,
                "oak_type": vinification.oak_type if vinification else None,
                "new_oak_percent": vinification.new_oak_percent if vinification else None,
                "aging_months": vinification.aging_months if vinification else None,
                "malolactic_fermentation": vinification.malolactic_fermentation if vinification else None,
                "filtration_notes": vinification.filtration_notes if vinification else None
            } if vinification else None,
            "vintages": vintages_data
        }

    finally:
        db.close()


@app.get("/api/journal")
def list_journal_entries():
    """
    Returns all journal entries newest-first, linking to wine identity and
    including predicted vs actual tasting comparison data.
    """
    db = SessionLocal()
    try:
        entries = (
            db.scalars(
                select(JournalEntry)
                .options(
                    joinedload(JournalEntry.tasting_profile),
                    joinedload(JournalEntry.vintage)
                    .joinedload(Vintage.wine)
                    .joinedload(Wine.producer)
                )
                .order_by(desc(JournalEntry.drank_at))
            )
            .unique()
            .all()
        )

        results = []
        for e in entries:
            v = e.vintage
            w = v.wine
            p = w.producer
            tp = e.tasting_profile
            results.append({
                "id": e.id,
                "drank_at": e.drank_at.isoformat() if e.drank_at else None,
                "location_name": e.location_name,
                "occasion": e.occasion,
                "companions": e.companions,
                "price_paid": e.price_paid,
                "personal_rating": e.personal_rating,
                "user_notes": e.user_notes,
                "label_photo_path": e.label_photo_path,
                "wine": {
                    "id": w.id,
                    "name": w.name,
                    "wine_type": w.wine_type.value if hasattr(w.wine_type, "value") else str(w.wine_type),
                    "appellation": w.appellation,
                    "producer_name": p.name,
                    "region": p.region
                },
                "vintage": {
                    "id": v.id,
                    "year": v.year,
                    "predicted_profile": v.predicted_profile_json
                },
                "tasting_profile": {
                    "color_hue": tp.color_hue if tp else None,
                    "nose_intensity": tp.nose_intensity.value if tp and tp.nose_intensity else None,
                    "palate_sweetness": tp.palate_sweetness if tp else None,
                    "palate_acidity": tp.palate_acidity.value if tp and tp.palate_acidity else None,
                    "palate_tannin": tp.palate_tannin.value if tp and tp.palate_tannin else None,
                    "palate_body": tp.palate_body.value if tp and tp.palate_body else None,
                    "palate_finish": tp.palate_finish.value if tp and tp.palate_finish else None,
                    "overall_quality": tp.overall_quality.value if tp and tp.overall_quality else None,
                    "aroma_descriptors": tp.aroma_descriptors if tp else {}
                } if tp else None
            })

        return results
    finally:
        db.close()


# ---------------------------------------------------------------------------
# API Endpoints: Personal Tasting Journal & Chat Sommelier
# ---------------------------------------------------------------------------

@app.post("/api/journal/save-entry")
def save_journal_entry(entry: TastingEntryCreate):
    """Saves personal tasting experience, geolocation, and WSET SAT sensory metrics."""
    db = SessionLocal()
    try:
        journal_entry = JournalEntry(
            vintage_id=entry.vintage_id,
            drank_at=datetime.fromisoformat(entry.drank_at) if entry.drank_at else datetime.utcnow(),
            location_name=entry.location_name,
            latitude=entry.latitude,
            longitude=entry.longitude,
            occasion=entry.occasion,
            companions=entry.companions,
            price_paid=entry.price_paid,
            personal_rating=entry.personal_rating,
            user_notes=entry.user_notes,
            label_photo_path=entry.label_photo_path
        )
        db.add(journal_entry)
        db.flush()

        tasting = TastingProfile(
            journal_entry_id=journal_entry.id,
            color_hue=entry.color_hue,
            nose_intensity=entry.nose_intensity,
            palate_sweetness=entry.palate_sweetness,
            palate_acidity=entry.palate_acidity,
            palate_tannin=entry.palate_tannin,
            palate_body=entry.palate_body,
            palate_finish=entry.palate_finish,
            overall_quality=entry.overall_quality
        )
        db.add(tasting)
        db.commit()

        return {"status": "success", "journal_entry_id": journal_entry.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/api/chat")
async def chat_with_wine_assistant(query: ChatQuery):
    """
    Conversational AI grounded in the wine's deep research dossier and user tasting notes.
    """
    db = SessionLocal()
    try:
        corpus = db.scalar(select(ResearchCorpus).where(ResearchCorpus.vintage_id == query.vintage_id))
        entry = db.scalar(select(JournalEntry).where(JournalEntry.vintage_id == query.vintage_id))

        context_str = "No prior research available."
        if corpus:
            context_str = corpus.full_research_markdown

        notes_str = "No personal tasting notes logged yet."
        if entry:
            notes_str = f"Rating: {entry.personal_rating}/10, Location: {entry.location_name}, Notes: {entry.user_notes}"

        prompt = f"""
        You are an expert AI Sommelier assistant for this specific wine bottle.
        
        === WINE RESEARCH DOSSIER ===
        {context_str}
        
        === USER'S PERSONAL TASTING EXPERIENCE ===
        {notes_str}
        
        === USER QUESTION ===
        {query.user_message}
        
        Answer the user's question clearly and accurately using the research context above.
        """

        response = research_agent.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return {"response": response.text}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Frontend Single-Page App
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the single-page web UI."""
    index_path = Path("static/index.html")
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Web Frontend missing! Place index.html inside /static folder.</h1>"
