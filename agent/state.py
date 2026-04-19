from typing import TypedDict, Optional, List, Dict, Any

class PropertyFeatures(TypedDict):
    """Input property features from the user."""
    overall_qual: int
    gr_liv_area: float
    garage_area: float
    first_flr_sf: float
    year_built: int
    full_bath: int
    year_remod: int
    mas_vnr_area: float
    fireplaces: int
    bsmt_fin_sf1: float
    lot_frontage: float
    wood_deck_sf: float
    open_porch_sf: float
    bsmt_full_bath: float
    paved_drive: int
    lot_area: float
    central_air: int
    roof_style: int


class ComparableProperty(TypedDict):
    """A comparable property from the dataset."""
    sale_price: float
    gr_liv_area: float
    overall_qual: int
    year_built: int
    garage_area: float
    full_bath: int
    similarity_score: float


class AgentState(TypedDict):
    """
    Central state object passed between all LangGraph nodes.
    Each node reads from and writes to this shared state.
    """
    # Input
    property_features: PropertyFeatures

    # Node 1: Price Predictor
    predicted_price: Optional[float]
    price_confidence: Optional[str]      
    price_range_low: Optional[float]
    price_range_high: Optional[float]

    # Node 2: Market Researcher (RAG)
    market_insights: Optional[List[str]]  # Retrieved RAG chunks
    market_summary: Optional[str]         # LLM-synthesised summary

    # Node 3: Comparables Analyzer
    comparables: Optional[List[ComparableProperty]]
    comps_summary: Optional[str]

    # Node 4: Advisory Generator
    recommendation: Optional[str]         
    investment_score: Optional[float]     
    reasoning: Optional[str]

    # Node 5: Report Formatter 
    final_report: Optional[Dict[str, str]]  

    # Control
    error: Optional[str]
    current_step: Optional[str]
