"""
Autonomous Deep-Research Agent for Wine & Terroir Analysis
Executes parallel multi-domain search tasks using Gemini Grounding & Search Tools.
"""

import os
import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Structured Research Models
# ---------------------------------------------------------------------------

class ResearchDomainResult(BaseModel):
    domain_name: str
    summary_md: str
    sources_found: list[str] = Field(default_factory=list)


class DeepWineResearchReport(BaseModel):
    producer_philosophy: ResearchDomainResult
    terroir_and_climate: ResearchDomainResult
    vinification_and_aging: ResearchDomainResult
    appellation_and_varietal: ResearchDomainResult
    full_compiled_markdown: str


# ---------------------------------------------------------------------------
# Autonomous Agent Engine
# ---------------------------------------------------------------------------

class WineResearchAgent:
    """Agent that conducts parallel deep dives into winery, terroir, vinification, and appellation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or explicit key required.")
        self.client = genai.Client(api_key=self.api_key)

    async def _research_domain(
        self, domain_name: str, prompt: str, search_query: str
    ) -> ResearchDomainResult:
        """Executes a search-grounded research query for a single domain in a separate thread."""
        print(f"🔬 [Agent Domain Task] Starting research on: {domain_name}...")

        full_prompt = f"""
        You are a master sommelier and viticulture researcher conducting an exhaustive research task.
        Use web search to find verified details about this wine.
        
        Search hint / query: {search_query}
        
        Task:
        {prompt}
        
        Provide a detailed, beautifully formatted Markdown report with sections, facts, and technical details.
        Avoid vague generalities—include specific soil types, oak percentages, winemaker quotes, or harvest dates where available.
        """

        # Wrap synchronous Gemini call in asyncio executor for true concurrent execution
        def call_gemini():
            return self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2,
                ),
            )

        response = await asyncio.to_thread(call_gemini)

        # Extract search grounding URLs if present
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, "web") and chunk.web and chunk.web.uri:
                        sources.append(chunk.web.uri)

        sources = list(set(sources))  # Deduplicate sources

        print(f"✅ [Agent Domain Task] Completed domain: {domain_name} ({len(sources)} sources cited)")
        
        return ResearchDomainResult(
            domain_name=domain_name,
            summary_md=response.text or "No detailed data found.",
            sources_found=sources
        )

    async def conduct_deep_research(
        self,
        producer: str,
        wine_name: str,
        vintage: Optional[int] = None,
        region: Optional[str] = None,
        appellation: Optional[str] = None
    ) -> DeepWineResearchReport:
        """
        Launches 4 parallel async research tasks across all core wine dimensions.
        """
        vintage_str = str(vintage) if vintage else "Non-Vintage"
        wine_identifier = f"{producer} {wine_name} {vintage_str}".strip()
        print(f"\n🚀 Launching Autonomous Agentic Deep-Research for: '{wine_identifier}'\n")

        # Define 4 distinct domain search prompts
        task_1 = self._research_domain(
            domain_name="Producer & Winemaker Philosophy",
            prompt=f"Research the winery '{producer}'. Detail its founding history, key winemakers, estate vs. negociant fruit, and farming philosophy (e.g., organic, biodynamic, sustainable, or conventional practices).",
            search_query=f"{producer} winery history winemaker biodynamic philosophy"
        )

        task_2 = self._research_domain(
            domain_name="Terroir, Soil & Vintage Climate",
            prompt=f"Analyze the terroir for '{wine_identifier}' in {region or appellation or 'its region'}. Find soil types (e.g., marl, limestone, schist, gravel), vineyard elevation, aspect/exposure, and microclimate. What were the specific growing conditions and weather patterns for the {vintage_str} harvest year?",
            search_query=f"{producer} {wine_name} {vintage_str} terroir soil weather harvest report"
        )

        task_3 = self._research_domain(
            domain_name="Viticulture & Vinification Tech Sheet",
            prompt=f"Examine the technical winemaking sheet for '{wine_identifier}'. Extract harvest details, yeast selection (ambient/wild vs cultured), fermentation vessels (oak, concrete, stainless steel), malolactic fermentation, oak aging regimen (% new oak, barrel type, duration in months), and filtration notes.",
            search_query=f"{producer} {wine_name} {vintage_str} tech sheet vinification oak aging fermentation"
        )

        task_4 = self._research_domain(
            domain_name="Appellation Rules & Grape Varietal Context",
            prompt=f"Investigate the grape varietal blend and legal appellation laws for '{wine_identifier}' ({appellation or region}). Detail the exact percentage blend if applicable, legal classification rules (DOCG, Grand Cru, AVA), and classical aging potential.",
            search_query=f"{producer} {wine_name} grape blend percentage {appellation or region} classification"
        )

        # Run all 4 tasks concurrently using asyncio.gather
        p_res, t_res, v_res, a_res = await asyncio.gather(task_1, task_2, task_3, task_4)

        # Compile master Markdown document
        compiled_markdown = f"""# Deep Research Dossier: {wine_identifier}

## 1. Winery & Winemaker Philosophy
{p_res.summary_md}

---

## 2. Terroir, Vineyard Geography & Vintage Conditions
{t_res.summary_md}

---

## 3. Vinification & Technical Winemaking
{v_res.summary_md}

---

## 4. Appellation Context & Grape Blend
{a_res.summary_md}

---

## Verified Research Sources
"""
        all_sources = set(p_res.sources_found + t_res.sources_found + v_res.sources_found + a_res.sources_found)
        for src in sorted(all_sources):
            compiled_markdown += f"- [{src}]({src})\n"

        return DeepWineResearchReport(
            producer_philosophy=p_res,
            terroir_and_climate=t_res,
            vinification_and_aging=v_res,
            appellation_and_varietal=a_res,
            full_compiled_markdown=compiled_markdown
        )


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def main():
        agent = WineResearchAgent()
        report = await agent.conduct_deep_research(
            producer="Tenuta San Guido",
            wine_name="Sassicaia",
            vintage=2018,
            region="Tuscany",
            appellation="Bolgheri Sassicaia DOC"
        )
        print("\n================ COMPILED RESEARCH REPORT ================\n")
        print(report.full_compiled_markdown[:2000] + "\n\n[... truncated for preview ...]")

    asyncio.run(main())