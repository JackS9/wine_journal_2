"""
Label Vision OCR & Vivino Structural Data Extraction Pipeline
Requires: google-genai, requests, beautifulsoup4, pydantic, Pillow
"""

import os
import re
import json
from typing import Optional, Dict, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Pipeline Outputs
# ---------------------------------------------------------------------------

class LabelExtractionResult(BaseModel):
    producer: str = Field(description="Name of the winery or producer")
    wine_name: str = Field(description="Specific name/label of the wine")
    vintage: Optional[int] = Field(default=None, description="Vintage year, if present")
    region: Optional[str] = Field(default=None, description="Region or state (e.g., Napa Valley, Piedmont)")
    country: Optional[str] = Field(default=None, description="Country of origin")
    appellation: Optional[str] = Field(default=None, description="Specific appellation or DOCG/AOP")
    wine_type: str = Field(default="red", description="red, white, rose, sparkling, fortified, or dessert")


class VivinoTasteStructure(BaseModel):
    boldness: Optional[float] = Field(default=None, description="Boldness scale 0.0 to 5.0")
    tannin: Optional[float] = Field(default=None, description="Tannin scale 0.0 to 5.0")
    sweetness: Optional[float] = Field(default=None, description="Sweetness scale 0.0 to 5.0")
    acidity: Optional[float] = Field(default=None, description="Acidity scale 0.0 to 5.0")


class VivinoExtractedData(BaseModel):
    vivino_id: Optional[str] = None
    page_url: Optional[str] = None
    rating_average: Optional[float] = None
    ratings_count: Optional[int] = None
    price_usd: Optional[float] = None
    structure: VivinoTasteStructure = Field(default_factory=VivinoTasteStructure)
    top_flavor_notes: list[str] = Field(default_factory=list)
    raw_next_data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 1. Gemini Vision Label OCR & Entity Normalizer
# ---------------------------------------------------------------------------

class LabelVisionAnalyzer:
    """Uses Gemini Vision API to parse wine labels into normalized JSON."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or explicit key required.")
        self.client = genai.Client(api_key=self.api_key)

    def analyze_label(self, image_path: str) -> LabelExtractionResult:
        """Parses a wine label photo and returns structured wine entity metadata."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Label image not found at {image_path}")

        image = Image.open(path)

        prompt = """
        Examine this wine label image carefully. Extract the key wine details into a structured JSON response.
        Ensure winery/producer, wine title, vintage year, region, country, and appellation are accurately captured.
        """

        response = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LabelExtractionResult,
                temperature=0.1
            )
        )

        return LabelExtractionResult.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# 2. Vivino Embedded JavaScript Scraper & Parser
# ---------------------------------------------------------------------------

class VivinoDataExtractor:
    """Queries Vivino, extracts HTML, and parses inline Next.js __NEXT_DATA__ JSON."""

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def search_and_fetch_wine_page(self, query: str) -> Optional[tuple[str, str]]:
        """
        Searches Vivino for a wine query and returns (url, html_content).
        """
        search_url = f"https://www.vivino.com/search/wines?q={requests.utils.quote(query)}"
        try:
            resp = self.session.get(search_url, timeout=10)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            # Find the first wine card link in Vivino search results
            first_link = soup.select_one("a[href*='/wines/'], a[href*='/wine/']")
            if not first_link or not first_link.get("href"):
                return None

            wine_path = first_link["href"]
            if not wine_path.startswith("http"):
                wine_url = f"https://www.vivino.com{wine_path}"
            else:
                wine_url = wine_path

            wine_page_resp = self.session.get(wine_url, timeout=10)
            if wine_page_resp.status_code == 200:
                return (wine_url, wine_page_resp.text)
        except Exception as e:
            print(f"[VivinoExtractor] Error fetching search results: {e}")
        return None

    def parse_next_data(self, html_content: str, page_url: Optional[str] = None) -> VivinoExtractedData:
        """
        Parses embedded <script id="__NEXT_DATA__"> or window.__PRELOADED_STATE__ from Vivino HTML.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        result = VivinoExtractedData(page_url=page_url)

        # 1. Locate Next.js JSON state script tag
        next_data_script = soup.find("script", id="__NEXT_DATA__")
        raw_json = None

        if next_data_script and next_data_script.string:
            try:
                raw_json = json.loads(next_data_script.string)
            except json.JSONDecodeError:
                pass

        # 2. Fallback to regex search for preloaded state if __NEXT_DATA__ isn't found
        if not raw_json:
            match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*({.*?});</script>", html_content, re.DOTALL)
            if match:
                try:
                    raw_json = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        if not raw_json:
            print("[VivinoExtractor] Warning: Could not locate JS data block.")
            return result

        result.raw_next_data = raw_json

        # 3. Traverse Next.js pageProps state to extract taste structure & ratings
        try:
            page_props = raw_json.get("props", {}).get("pageProps", {})
            wine_data = page_props.get("wine", {}) or page_props.get("vintage", {}).get("wine", {})

            result.vivino_id = str(wine_data.get("id", ""))
            
            # Extract Ratings & Pricing
            stats = wine_data.get("statistics", {})
            result.rating_average = stats.get("ratings_average")
            result.ratings_count = stats.get("ratings_count")

            price = page_props.get("price", {}) or wine_data.get("price", {})
            if isinstance(price, dict):
                result.price_usd = price.get("amount")

            # Extract 5-point Structural Scales (Taste Profile)
            taste = wine_data.get("taste", {}) or page_props.get("taste", {})
            structure = taste.get("structure", {})
            if structure:
                result.structure = VivinoTasteStructure(
                    boldness=structure.get("boldness"),
                    tannin=structure.get("tannin"),
                    sweetness=structure.get("sweetness"),
                    acidity=structure.get("acidity")
                )

            # Extract Community Flavor Notes
            flavor_groups = taste.get("flavor", [])
            flavors = []
            if isinstance(flavor_groups, list):
                for group in flavor_groups:
                    group_name = group.get("group")
                    stats_info = group.get("stats", {})
                    if group_name and stats_info.get("count", 0) > 0:
                        flavors.append(group_name)
            result.top_flavor_notes = flavors[:10]

        except Exception as e:
            print(f"[VivinoExtractor] Error extracting structural fields: {e}")

        return result


# ---------------------------------------------------------------------------
# 3. Master Pipeline Orchestrator
# ---------------------------------------------------------------------------

class WineLabelIngestionPipeline:
    """Combines Label OCR with Vivino Scraping for an end-to-end extraction workflow."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.vision_analyzer = LabelVisionAnalyzer(api_key=gemini_api_key)
        self.vivino_extractor = VivinoDataExtractor()

    def process_image(self, image_path: str) -> tuple[LabelExtractionResult, VivinoExtractedData]:
        """Runs image through OCR, searches Vivino, and returns both structured outputs."""
        print(f"📷 Analyzing bottle label: {image_path}...")
        label_info = self.vision_analyzer.analyze_label(image_path)
        
        # Build search query string
        query = f"{label_info.producer} {label_info.wine_name}"
        if label_info.vintage:
            query += f" {label_info.vintage}"

        print(f"🔍 Querying Vivino for: '{query}'...")
        vivino_res = self.vivino_extractor.search_and_fetch_wine_page(query)

        if vivino_res:
            page_url, html_content = vivino_res
            print(f"🔗 Found Vivino page: {page_url}")
            vivino_data = self.vivino_extractor.parse_next_data(html_content, page_url=page_url)
        else:
            print("⚠️ Vivino page not found; returning vision data only.")
            vivino_data = VivinoExtractedData()

        return label_info, vivino_data


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is set in your environment
    pipeline = WineLabelIngestionPipeline()
    
    # Test run with a label image
    test_image = "test_wine_label.jpg"
    if Path(test_image).exists():
        label_data, vivino_data = pipeline.process_image(test_image)
        print("\n--- VISION OCR RESULT ---")
        print(label_data.model_dump_json(indent=2))
        print("\n--- VIVINO STRUCTURED METRICS ---")
        print(vivino_data.model_dump_json(indent=2, exclude={"raw_next_data"}))