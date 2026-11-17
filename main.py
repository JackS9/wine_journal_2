"""
FastAPI Application Backend for AI Wine Journal
Handles label image uploads, agent streaming research (SSE), database persistence, and conversational Q&A.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Import modules built in previous steps
from models import Base, Producer, Wine, Vintage, VivinoMetadata, ResearchCorpus, JournalEntry, TastingProfile
from vivino_pipeline import WineLabelIngestionPipeline
from research_agent import WineResearchAgent

# Initialize FastAPI App
app = FastAPI(title="AI Agentic Wine Journal", version="2.0")

# Setup SQLite Database Engine (or change to PostgreSQL if desired)
DB_PATH = "wine_journal.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pipeline instances
pipeline = WineLabelIngestionPipeline()
research_agent = WineResearchAgent()

# Static files for web frontend
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


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


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/journal/scan")
async def scan_label_and_initiate_research(file: UploadFile = File(...)):
    """
    Accepts label photo, runs Vision OCR + Vivino scraping, creates DB entities,
    and returns a vintage_id for streaming agent research.
    """
    # 1. Save uploaded image to disk
    file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = UPLOAD_DIR / f"label_{timestamp}.{file_ext}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Run Vision OCR & Vivino pipeline
    label_info, vivino_data = pipeline.process_image(str(file_path))

    # 3. Store in Database
    db = SessionLocal()
    try:
        # Producer
        producer = db.scalar(select(Producer).where(Producer.name == label_info.producer))
        if not producer:
            producer = Producer(
                name=label_info.producer,
                country=label_info.country or "Unknown",
                region=label_info.region or "Unknown"
            )
            db.add(producer)
            db.flush()

        # Wine
        wine = db.scalar(select(Wine).where(Wine.producer_id == producer.id, Wine.name == label_info.wine_name))
        if not wine:
            wine = Wine(
                producer_id=producer.id,
                name=label_info.wine_name,
                wine_type=label_info.wine_type.lower(),
                appellation=label_info.appellation
            )
            db.add(wine)
            db.flush()

        # Vintage
        vintage = db.scalar(select(Vintage).where(Vintage.wine_id == wine.id, Vintage.year == label_info.vintage))
        if not vintage:
            vintage = Vintage(wine_id=wine.id, year=label_info.vintage)
            db.add(vintage)
            db.flush()

        # Vivino Metadata
        if vivino_data.rating_average or vivino_data.structure.boldness:
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

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

    return {
        "vintage_id": vintage_id,
        "image_url": f"/uploads/{file_path.name}",
        "ocr_extracted": label_info.model_dump(),
        "vivino_data": vivino_data.model_dump(exclude={"raw_next_data"})
    }


@app.get("/api/journal/research-stream/{vintage_id}")
async def research_agent_stream(vintage_id: int):
    """
    Streams agent research progress in real-time via Server-Sent Events (SSE).
    """
    db = SessionLocal()
    vintage = db.get(Vintage, vintage_id)
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
        yield f"data: {json.dumps({'status': 'started', 'message': f'Launching parallel AI research agent for {producer_name} {wine_name}...'})}\n\n"
        await asyncio.sleep(0.5)

        # Run deep research agent
        yield f"data: {json.dumps({'status': 'agent_running', 'message': 'Searching winery history, soil profiles, vintage weather, and tech sheets...'})}\n\n"
        
        report = await research_agent.conduct_deep_research(
            producer=producer_name,
            wine_name=wine_name,
            vintage=year,
            region=region,
            appellation=appellation
        )

        # Save research report to database
        db_session = SessionLocal()
        corpus = ResearchCorpus(
            vintage_id=vintage_id,
            winery_history_md=report.producer_philosophy.summary_md,
            terroir_deep_dive_md=report.terroir_and_climate.summary_md,
            vintage_weather_report_md=report.terroir_and_climate.summary_md,
            winemaking_tech_sheet_md=report.vinification_and_aging.summary_md,
            full_research_markdown=report.full_compiled_markdown
        )
        db_session.add(corpus)
        db_session.commit()
        db_session.close()

        yield f"data: {json.dumps({'status': 'completed', 'markdown': report.full_compiled_markdown})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
            user_notes=entry.user_notes
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
    corpus = db.scalar(select(ResearchCorpus).where(ResearchCorpus.vintage_id == query.vintage_id))
    entry = db.scalar(select(JournalEntry).where(JournalEntry.vintage_id == query.vintage_id))

    context_str = "No prior research available."
    if corpus:
        context_str = corpus.full_research_markdown

    notes_str = "No personal tasting notes logged yet."
    if entry:
        notes_str = f"Rating: {entry.personal_rating}/10, Location: {entry.location_name}, Notes: {entry.user_notes}"

    db.close()

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

    # Use Gemini client directly
    response = research_agent.client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return {"response": response.text}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the single-page web UI."""
    index_path = Path("static/index.html")
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Web Frontend missing! Place index.html inside /static folder.</h1>"