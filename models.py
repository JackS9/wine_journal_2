"""
Wine Journal Database Schema & Ontology Definitions
Compatible with SQLAlchemy 2.0, Pydantic v2, WSET SAT Standards, and Schema.org
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    String, Text, Float, Integer, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pydantic import BaseModel, Field, HttpUrl

# ---------------------------------------------------------------------------
# SQLAlchemy Base & Enums
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass

class WineType(str, PyEnum):
    RED = "red"
    WHITE = "white"
    ROSE = "rose"
    SPARKLING = "sparkling"
    FORTIFIED = "fortified"
    DESSERT = "dessert"

class FarmingPractice(str, PyEnum):
    CONVENTIONAL = "conventional"
    ORGANIC = "organic"
    BIODYNAMIC = "biodynamic"
    NATURAL = "natural"
    SUSTAINABLE = "sustainable"

class WSETIntensity(str, PyEnum):
    LIGHT = "light"
    MEDIUM_MINUS = "medium_minus"
    MEDIUM = "medium"
    MEDIUM_PLUS = "medium_plus"
    PRONOUNCED = "pronounced"

class WSETQuality(str, PyEnum):
    POOR = "poor"
    ACCEPTABLE = "acceptable"
    GOOD = "good"
    VERY_GOOD = "very_good"
    OUTSTANDING = "outstanding"

# ---------------------------------------------------------------------------
# ORM Entities
# ---------------------------------------------------------------------------

class Producer(Base):
    """Represents the winery or producer."""
    __tablename__ = "producers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    canonical_name: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    region: Mapped[str] = mapped_column(String(150), nullable=False)
    subregion: Mapped[Optional[str]] = mapped_column(String(150))
    farming_practices: Mapped[Optional[FarmingPractice]] = mapped_column(SQLEnum(FarmingPractice))
    certifications: Mapped[Optional[List[str]]] = mapped_column(JSON)
    website_url: Mapped[Optional[str]] = mapped_column(String(500))
    winemaker_philosophy: Mapped[Optional[str]] = mapped_column(Text)
    history_md: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    wines: Mapped[List["Wine"]] = relationship("Wine", back_populates="producer", cascade="all, delete-orphan")


class Wine(Base):
    """Represents a specific wine brand/label produced by a winery."""
    __tablename__ = "wines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    producer_id: Mapped[int] = mapped_column(ForeignKey("producers.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    wine_type: Mapped[WineType] = mapped_column(SQLEnum(WineType), nullable=False)
    classification: Mapped[Optional[str]] = mapped_column(String(150))  # e.g., Grand Cru, DOCG, AVA
    appellation: Mapped[Optional[str]] = mapped_column(String(150))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    producer: Mapped["Producer"] = relationship("Producer", back_populates="wines")
    vintages: Mapped[List["Vintage"]] = relationship("Vintage", back_populates="wine", cascade="all, delete-orphan")
    terroir: Mapped[Optional["Terroir"]] = relationship("Terroir", uselist=False, back_populates="wine", cascade="all, delete-orphan")
    vinification: Mapped[Optional["Vinification"]] = relationship("Vinification", uselist=False, back_populates="wine", cascade="all, delete-orphan")


class GrapeVarietal(Base):
    """Grape varietal master record."""
    __tablename__ = "grape_varietals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    aliases: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # e.g., ["Syrah", "Shiraz"]


class WineBlend(Base):
    """Junction table for wine vintage grape blend composition."""
    __tablename__ = "wine_blends"

    vintage_id: Mapped[int] = mapped_column(ForeignKey("vintages.id"), primary_key=True)
    varietal_id: Mapped[int] = mapped_column(ForeignKey("grape_varietals.id"), primary_key=True)
    percentage: Mapped[Optional[float]] = mapped_column(Float)

    vintage: Mapped["Vintage"] = relationship("Vintage", back_populates="blends")
    varietal: Mapped["GrapeVarietal"] = relationship("GrapeVarietal")


class Vintage(Base):
    """Represents a specific harvest year for a wine."""
    __tablename__ = "vintages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wine_id: Mapped[int] = mapped_column(ForeignKey("wines.id"), nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer, index=True)  # None for Non-Vintage (NV)
    abv: Mapped[Optional[float]] = mapped_column(Float)
    production_volume_cases: Mapped[Optional[int]] = mapped_column(Integer)
    drinking_window_start: Mapped[Optional[int]] = mapped_column(Integer)
    drinking_window_end: Mapped[Optional[int]] = mapped_column(Integer)
    weather_summary: Mapped[Optional[str]] = mapped_column(Text)
    predicted_profile_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    wine: Mapped["Wine"] = relationship("Wine", back_populates="vintages")
    blends: Mapped[List["WineBlend"]] = relationship("WineBlend", back_populates="vintage", cascade="all, delete-orphan")
    vivino_data: Mapped[Optional["VivinoMetadata"]] = relationship("VivinoMetadata", uselist=False, back_populates="vintage")
    research_corpus: Mapped[Optional["ResearchCorpus"]] = relationship("ResearchCorpus", uselist=False, back_populates="vintage")
    journal_entries: Mapped[List["JournalEntry"]] = relationship("JournalEntry", back_populates="vintage")
    proposals: Mapped[List["DossierProposal"]] = relationship("DossierProposal", back_populates="vintage", cascade="all, delete-orphan")


class Terroir(Base):
    """Vineyard microclimate, soil, and geography associated with the wine parcels."""
    __tablename__ = "terroirs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wine_id: Mapped[int] = mapped_column(ForeignKey("wines.id"), unique=True, nullable=False)
    single_vineyard_name: Mapped[Optional[str]] = mapped_column(String(150))
    soil_types: Mapped[Optional[List[str]]] = mapped_column(JSON)  # ["limestone", "marne", "gravel"]
    elevation_meters: Mapped[Optional[float]] = mapped_column(Float)
    aspect: Mapped[Optional[str]] = mapped_column(String(50))  # e.g., "South-facing 15% slope"

    wine: Mapped["Wine"] = relationship("Wine", back_populates="terroir")


class Vinification(Base):
    """Winemaking techniques, fermentation, and aging details for the cuvée."""
    __tablename__ = "vinifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wine_id: Mapped[int] = mapped_column(ForeignKey("wines.id"), unique=True, nullable=False)
    fermentation_vessel: Mapped[Optional[str]] = mapped_column(String(100))  # Stainless, Concrete egg, Oak
    yeast_type: Mapped[Optional[str]] = mapped_column(String(50))  # Ambient / Wild vs Cultured
    oak_type: Mapped[Optional[str]] = mapped_column(String(100))  # French, American, Slavonian
    new_oak_percent: Mapped[Optional[float]] = mapped_column(Float)
    aging_months: Mapped[Optional[int]] = mapped_column(Integer)
    malolactic_fermentation: Mapped[Optional[bool]] = mapped_column(Boolean)
    filtration_notes: Mapped[Optional[str]] = mapped_column(String(255))

    wine: Mapped["Wine"] = relationship("Wine", back_populates="vinification")


class VivinoMetadata(Base):
    """Data parsed from Vivino's embedded JavaScript structures."""
    __tablename__ = "vivino_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vintage_id: Mapped[int] = mapped_column(ForeignKey("vintages.id"), unique=True, nullable=False)
    vivino_id: Mapped[str] = mapped_column(String(100), index=True)
    page_url: Mapped[Optional[str]] = mapped_column(String(500))
    rating_average: Mapped[Optional[float]] = mapped_column(Float)
    ratings_count: Mapped[Optional[int]] = mapped_column(Integer)
    price_usd: Mapped[Optional[float]] = mapped_column(Float)
    
    # Vivino 5-point structure scales (0.0 to 5.0)
    boldness: Mapped[Optional[float]] = mapped_column(Float)
    tannin: Mapped[Optional[float]] = mapped_column(Float)
    sweetness: Mapped[Optional[float]] = mapped_column(Float)
    acidity: Mapped[Optional[float]] = mapped_column(Float)
    
    raw_js_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # Full parsed JS object

    vintage: Mapped["Vintage"] = relationship("Vintage", back_populates="vivino_data")


class ResearchCorpus(Base):
    """Deep-dive research collected by the autonomous agentic research engine."""
    __tablename__ = "research_corpora"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vintage_id: Mapped[int] = mapped_column(ForeignKey("vintages.id"), unique=True, nullable=False)
    winery_history_md: Mapped[Optional[str]] = mapped_column(Text)
    terroir_deep_dive_md: Mapped[Optional[str]] = mapped_column(Text)
    vintage_weather_report_md: Mapped[Optional[str]] = mapped_column(Text)
    winemaking_tech_sheet_md: Mapped[Optional[str]] = mapped_column(Text)
    full_research_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[Optional[List[str]]] = mapped_column(JSON)
    provenance: Mapped[Optional[str]] = mapped_column(String(50), default="label_scan")
    last_researched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    vintage: Mapped["Vintage"] = relationship("Vintage", back_populates="research_corpus")


class DossierProposal(Base):
    """Structured extraction proposal awaiting human-in-the-loop review."""
    __tablename__ = "dossier_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vintage_id: Mapped[int] = mapped_column(ForeignKey("vintages.id"), nullable=False, index=True)
    proposal_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)  # pending, accepted, rejected
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    vintage: Mapped["Vintage"] = relationship("Vintage", back_populates="proposals")


class JournalEntry(Base):
    """User's personal tasting experience log."""
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vintage_id: Mapped[int] = mapped_column(ForeignKey("vintages.id"), nullable=False)
    drank_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    location_name: Mapped[Optional[str]] = mapped_column(String(255))  # Venue or home
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    occasion: Mapped[Optional[str]] = mapped_column(String(255))
    companions: Mapped[Optional[str]] = mapped_column(String(255))
    price_paid: Mapped[Optional[float]] = mapped_column(Float)
    bottle_size_ml: Mapped[int] = mapped_column(Integer, default=750)
    label_photo_path: Mapped[Optional[str]] = mapped_column(String(500))
    personal_rating: Mapped[Optional[float]] = mapped_column(Float)  # e.g., 1.0 to 10.0 or 100-pt
    user_notes: Mapped[Optional[str]] = mapped_column(Text)
    provenance: Mapped[Optional[str]] = mapped_column(String(50), default="label_scan")

    vintage: Mapped["Vintage"] = relationship("Vintage", back_populates="journal_entries")
    tasting_profile: Mapped[Optional["TastingProfile"]] = relationship("TastingProfile", uselist=False, back_populates="journal_entry", cascade="all, delete-orphan")


class TastingProfile(Base):
    """Structured sensory assessment based on WSET SAT rules."""
    __tablename__ = "tasting_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    journal_entry_id: Mapped[int] = mapped_column(ForeignKey("journal_entries.id"), unique=True, nullable=False)

    # Sensory Indicators
    clarity: Mapped[Optional[str]] = mapped_column(String(50))  # Clear, Hazy
    color_hue: Mapped[Optional[str]] = mapped_column(String(50))  # Ruby, Garnet, Lemon, Gold
    nose_intensity: Mapped[Optional[WSETIntensity]] = mapped_column(SQLEnum(WSETIntensity))
    palate_sweetness: Mapped[Optional[str]] = mapped_column(String(50))  # Dry, Off-dry, Sweet
    palate_acidity: Mapped[Optional[WSETIntensity]] = mapped_column(SQLEnum(WSETIntensity))
    palate_tannin: Mapped[Optional[WSETIntensity]] = mapped_column(SQLEnum(WSETIntensity))
    palate_alcohol: Mapped[Optional[WSETIntensity]] = mapped_column(SQLEnum(WSETIntensity))
    palate_body: Mapped[Optional[WSETIntensity]] = mapped_column(SQLEnum(WSETIntensity))
    palate_finish: Mapped[Optional[WSETIntensity]] = mapped_column(SQLEnum(WSETIntensity))
    
    aroma_descriptors: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    overall_quality: Mapped[Optional[WSETQuality]] = mapped_column(SQLEnum(WSETQuality))

    journal_entry: Mapped["JournalEntry"] = relationship("JournalEntry", back_populates="tasting_profile")

# ---------------------------------------------------------------------------
# Schema.org JSON-LD Exporter (Global Wine App Interoperability)
# ---------------------------------------------------------------------------

class SchemaOrgWine(BaseModel):
    """Standardized JSON-LD representation matching Schema.org/Wine."""
    context: str = Field(default="https://schema.org", alias="@context")
    type: str = Field(default="Wine", alias="@type")
    name: str
    manufacturer: str
    description: Optional[str] = None
    wineType: Optional[str] = None
    alcoholByVolume: Optional[float] = None

    def to_json_ld(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)
