"""
Dossier Extractor Module
Converts unstructured research dossier markdown into a structured DossierExtraction model
using Gemini structured output (temperature 0.1).
"""

import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Dossier Extraction
# ---------------------------------------------------------------------------

class ProducerExtraction(BaseModel):
    canonical_name: Optional[str] = Field(None, description="Official/canonical estate name")
    country: Optional[str] = Field(None, description="Country of origin")
    region: Optional[str] = Field(None, description="Macro region (e.g. Bordeaux, Tuscany)")
    subregion: Optional[str] = Field(None, description="Subregion or district (e.g. Médoc, Left Bank)")
    farming_practices: Optional[str] = Field(
        None, 
        description="Coarse farming practice: 'conventional', 'organic', 'biodynamic', 'natural', or 'sustainable'"
    )
    certifications: List[str] = Field(
        default_factory=list, 
        description="List of specific certifications mentioned (e.g. ['HVE Level 3', 'Agriculture Biologique (AB)', 'Demeter'])"
    )
    winemaker_philosophy: Optional[str] = Field(
        None, 
        description="Summary of winemaking and viticultural philosophy (maximum 3 sentences)"
    )
    history_summary: Optional[str] = Field(
        None, 
        description="Concise summary of estate founding and ownership history (maximum 5 sentences)"
    )


class WineExtraction(BaseModel):
    classification: Optional[str] = Field(
        None, 
        description="Official classification (e.g. 'Grand Cru Classé (Fourth Growth)', 'DOCG', 'AVA', 'Premier Cru')"
    )
    appellation: Optional[str] = Field(
        None, 
        description="Appellation d'Origine / AVA (e.g. 'Margaux', 'Bolgheri')"
    )
    wine_type: Optional[str] = Field(
        None, 
        description="'red', 'white', 'rose', 'sparkling', 'fortified', or 'dessert'"
    )


class TerroirExtraction(BaseModel):
    soil_types: List[str] = Field(
        default_factory=list, 
        description="Short tags of distinct soil types (e.g. ['gravel', 'clay', 'sand', 'limestone'])"
    )
    elevation_meters: Optional[float] = Field(
        None, 
        description="Vineyard elevation in meters, if stated"
    )
    aspect: Optional[str] = Field(
        None, 
        description="Vineyard slope/exposure/aspect (e.g. 'South-facing slopes', 'Gentle slope toward the Gironde')"
    )
    single_vineyard_name: Optional[str] = Field(
        None, 
        description="Single vineyard or parcel name, if applicable"
    )


class VinificationExtraction(BaseModel):
    fermentation_vessel: Optional[str] = Field(
        None, 
        description="Fermentation vessels (e.g. 'Temperature-controlled stainless steel and concrete vats')"
    )
    yeast_type: Optional[str] = Field(
        None, 
        description="Yeast selection: ambient/indigenous/wild vs selected/cultured"
    )
    oak_type: Optional[str] = Field(
        None, 
        description="Type and origin of oak barrels (e.g. 'French oak barriques')"
    )
    new_oak_percent: Optional[float] = Field(
        None, 
        description="Percentage of new oak (0-100), if stated"
    )
    aging_months: Optional[int] = Field(
        None, 
        description="Barrel/cellar aging duration in months, if stated"
    )
    malolactic_fermentation: Optional[bool] = Field(
        None, 
        description="Whether malolactic fermentation occurred (true/false/null)"
    )
    filtration_notes: Optional[str] = Field(
        None, 
        description="Fining and filtration details (e.g. 'Unfined and unfiltered', 'Light egg-white fining')"
    )


class VintageExtraction(BaseModel):
    abv: Optional[float] = Field(
        None, 
        description="Alcohol by volume percentage (e.g. 13.5)"
    )
    drinking_window_start: Optional[int] = Field(
        None, 
        description="Beginning of optimal drinking window year (e.g. 2026)"
    )
    drinking_window_end: Optional[int] = Field(
        None, 
        description="End of optimal drinking window year (e.g. 2045)"
    )
    weather_summary: Optional[str] = Field(
        None, 
        description="Condensed summary of the vintage growing season and harvest weather conditions (maximum 4 sentences)"
    )


class BlendItem(BaseModel):
    varietal_name: str = Field(..., description="Canonical grape varietal name (e.g. 'Cabernet Sauvignon')")
    percentage: Optional[float] = Field(
        None, 
        description="Percentage in the final blend (0-100). If a range is stated, use the midpoint here and record the range in conflicts."
    )


class PredictedProfile(BaseModel):
    appearance: Optional[str] = Field(None, description="Visual appearance and color/hue notes")
    nose: List[str] = Field(default_factory=list, description="List of primary, secondary, or tertiary aromatic descriptors")
    palate: List[str] = Field(default_factory=list, description="Palate flavors, acidity, tannin quality, and body impressions")
    structure_notes: Optional[str] = Field(None, description="Summary of tannins, acidity, alcohol balance, and finish")
    cellaring_note: Optional[str] = Field(None, description="Sommelier aging potential and evolution note")


class ConflictItem(BaseModel):
    field: str = Field(..., description="Name of the field or concept where texts disagree")
    values_found: List[str] = Field(default_factory=list, description="All distinct values found across reports")
    note: Optional[str] = Field(None, description="Context on why or where the sources disagreed")


class DossierExtraction(BaseModel):
    producer: ProducerExtraction
    wine: WineExtraction
    terroir: TerroirExtraction
    vinification: VinificationExtraction
    vintage: VintageExtraction
    blend: List[BlendItem] = Field(default_factory=list)
    predicted_profile: PredictedProfile
    conflicts: List[ConflictItem] = Field(
        default_factory=list, 
        description="Disagreements between domain sections or sources (e.g. vineyard area ~40 vs 45-46 ha, HVE Level 3 vs 4, blend differences, planting specs)"
    )
    unmapped_facts: List[str] = Field(
        default_factory=list, 
        description="Salient facts with no explicit schema home (e.g. closure type, harvest dates, bottling date, vine age, planting density)"
    )


# ---------------------------------------------------------------------------
# Extractor Engine
# ---------------------------------------------------------------------------

class DossierExtractor:
    """Extracts structured wine knowledge from raw research markdown using Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or explicit key required.")
        self.client = genai.Client(api_key=self.api_key)

    def extract(
        self,
        corpus_markdown: str,
        producer_name: str,
        wine_name: str,
        vintage_year: Optional[int] = None
    ) -> DossierExtraction:
        vintage_str = str(vintage_year) if vintage_year else "Non-Vintage"
        prompt = f"""
You are an expert viticulture knowledge extraction engine.
You are given the full research dossier markdown for:
- Producer: {producer_name}
- Wine: {wine_name}
- Vintage: {vintage_str}

Your task is to parse this research text and extract precise structured data matching the schema.

STRICT EXTRACTION RULES:
1. Extract ONLY facts explicitly stated in the research text.
2. Return null (or empty list) for any field unstated or unknown.
3. NEVER infer, extrapolate, assume, or round values.
4. For percentages or numbers with ranges (e.g., blend "55-60% Cabernet Sauvignon" or aging "16 to 18 months"):
   - For blend percentage: use the arithmetic midpoint (e.g. 57.5) in the blend list AND record the stated range as a conflict in `conflicts`.
5. CONFLICT DETECTION IS MANDATORY:
   - Carefully cross-check the four domain sections in the dossier. When different sections or cited sources disagree, you MUST record every distinct value in `conflicts`.
   - Examples of common conflicts to check:
     * Vineyard area / estate size (e.g. ~39-40 ha planted vs 45-46 ha total estate).
     * Environmental certifications / HVE level (e.g. HVE Level 3 vs Level 4).
     * Varietal blend percentages (e.g. 60/35/5 vs 63/31/6).
     * Oak regime / aging months (e.g. 16-18 months vs 18 months, 40-50% vs 50% new oak).
   - Also conflict-check fields with no dedicated schema column (such as vineyard area, vine age, planting density). If sources disagree, record in `conflicts`. If consistent, fold them into the relevant summary or unmapped facts.
6. UNMAPPED FACTS:
   - Capture salient, high-value technical facts mentioned in the dossier that have no dedicated column (e.g., specific harvest dates, closure/cork type, bottling dates, average vine age, planting density, organic conversion year).
7. SUMMARY LENGTH CONSTRAINTS:
   - winemaker_philosophy: <= 3 sentences.
   - history_summary: <= 5 sentences.
   - weather_summary: <= 4 sentences.
8. PREDICTED PROFILE:
   - Extract the sommelier sensory tasting matrix (appearance, nose aromas, palate sensations, structural tannins/acid balance, cellaring potential) as predicted by the research literature.

RESEARCH DOSSIER TEXT:
\"\"\"
{corpus_markdown}
\"\"\"
"""

        response = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DossierExtraction,
                temperature=0.1,
            )
        )

        raw_json = response.text
        # Parse into DossierExtraction model
        extraction = DossierExtraction.model_validate_json(raw_json)
        return extraction


def extract_dossier_from_corpus(
    corpus_markdown: str,
    producer_name: str,
    wine_name: str,
    vintage_year: Optional[int] = None
) -> DossierExtraction:
    extractor = DossierExtractor()
    return extractor.extract(corpus_markdown, producer_name, wine_name, vintage_year)
