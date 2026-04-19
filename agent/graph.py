import os
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import predict_price, retrieve_market_insights, find_comparables

# LLM
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm


# Node 1: Price Predictor

def price_predictor_node(state: AgentState) -> Dict[str, Any]:
    """Run the ML model to get price estimate and confidence band."""
    try:
        result = predict_price(state["property_features"])
        return {
            "predicted_price":  result["predicted_price"],
            "price_range_low":  result["price_range_low"],
            "price_range_high": result["price_range_high"],
            "price_confidence": result["price_confidence"],
            "current_step":     "price_predictor",
        }
    except Exception as e:
        return {"error": f"Price prediction failed: {e}", "current_step": "price_predictor"}


# Node 2: Market Researcher (RAG)

def market_researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve relevant market knowledge via ChromaDB RAG,
    then ask the LLM to synthesise a concise market summary.
    """
    try:
        features = state["property_features"]
        price = state.get("predicted_price", 200000)
        query = (
            f"Property with overall quality {features.get('overall_qual')}, "
            f"built in {features.get('year_built')}, "
            f"living area {features.get('gr_liv_area')} sq ft, "
            f"estimated value ${price:,.0f}. "
            "Provide market insights, risk factors, and appreciation potential."
        )

        chunks = retrieve_market_insights(query, k=5)

        # Synthesise with LLM
        context = "\n\n---\n\n".join(chunks)
        prompt = f"""You are a real estate market analyst. Based on the following knowledge base excerpts, 
write a concise 3–4 sentence market summary for this property investment scenario.

PROPERTY CONTEXT:
- Estimated Value: ${price:,.0f}
- Overall Quality: {features.get('overall_qual')}/10
- Year Built: {features.get('year_built')}
- Living Area: {features.get('gr_liv_area'):,.0f} sq ft

KNOWLEDGE BASE CONTEXT:
{context}

Write your market summary below (3–4 sentences, factual, no disclaimers here):"""

        response = get_llm().invoke(prompt)
        summary = response.content.strip()

        return {
            "market_insights": chunks,
            "market_summary":  summary,
            "current_step":    "market_researcher",
        }
    except Exception as e:
        return {"error": f"Market research failed: {e}", "current_step": "market_researcher"}


# Node 3: Comparables Analyzer

def comparables_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Find similar sold properties in the Ames dataset,
    then generate a comparables analysis narrative.
    """
    try:
        features = state["property_features"]
        comps = find_comparables(features, n=5)
        predicted = state.get("predicted_price", 0)

        if comps:
            avg_comp_price = sum(c["sale_price"] for c in comps) / len(comps)
            price_vs_comps = predicted - avg_comp_price
            direction = "above" if price_vs_comps > 0 else "below"

            comp_lines = "\n".join([
                f"  - ${c['sale_price']:,.0f} | {c['gr_liv_area']:,.0f} sqft | "
                f"Qual {c['overall_qual']}/10 | Built {c['year_built']} | "
                f"Similarity: {c['similarity_score']:.2f}"
                for c in comps
            ])

            prompt = f"""You are a real estate appraiser. Analyze these comparable sales and write a 
2–3 sentence summary of how the subject property compares to recent sales.

SUBJECT PROPERTY ESTIMATE: ${predicted:,.0f}
COMPARABLE SALES (similar properties):
{comp_lines}

Average comp price: ${avg_comp_price:,.0f}
Subject is ${abs(price_vs_comps):,.0f} {direction} the comp average.

Write a brief, factual comparables analysis (2–3 sentences):"""

            response = get_llm().invoke(prompt)
            comps_summary = response.content.strip()
        else:
            comps_summary = "Insufficient comparable data available for analysis."

        return {
            "comparables":  comps,
            "comps_summary": comps_summary,
            "current_step": "comparables_analyzer",
        }
    except Exception as e:
        return {"error": f"Comparables analysis failed: {e}", "current_step": "comparables_analyzer"}


# Node 4: Advisory Generator

def advisory_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesise all gathered signals into a final BUY / HOLD / AVOID
    recommendation with an investment score and reasoning.
    """
    try:
        features = state["property_features"]
        predicted = state.get("predicted_price", 0)
        low = state.get("price_range_low", 0)
        high = state.get("price_range_high", 0)
        confidence = state.get("price_confidence", "Medium")
        market_summary = state.get("market_summary", "No market data available.")
        comps_summary = state.get("comps_summary", "No comparables available.")

        prompt = f"""You are a senior real estate investment advisor. 
Based on all available analysis, provide a structured investment recommendation.

PROPERTY DETAILS
Overall Quality: {features.get('overall_qual')}/10
Year Built: {features.get('year_built')}
Last Remodeled: {features.get('year_remod')}
Living Area: {features.get('gr_liv_area'):,.0f} sq ft
Lot Area: {features.get('lot_area'):,.0f} sq ft
Central Air: {"Yes" if features.get('central_air') == 1 else "No"}
Fireplaces: {features.get('fireplaces')}
Garage Area: {features.get('garage_area'):,.0f} sq ft

ML VALUATION
Predicted Price: ${predicted:,.0f}
Price Range: ${low:,.0f} – ${high:,.0f}
Model Confidence: {confidence}

MARKET ANALYSIS
{market_summary}

COMPARABLE SALES
{comps_summary}

Based on all the above, respond with EXACTLY this format:

RECOMMENDATION: [BUY | HOLD | AVOID]
INVESTMENT_SCORE: [1-10]
REASONING: [3–4 sentences explaining the recommendation, mentioning key risk factors and upsides]
"""

        response = get_llm().invoke(prompt)
        text = response.content.strip()

        # Parse structured output
        recommendation = "HOLD"
        investment_score = 5.0
        reasoning = text

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("RECOMMENDATION:"):
                rec = line.split(":", 1)[1].strip().upper()
                if "BUY" in rec:
                    recommendation = "BUY"
                elif "AVOID" in rec:
                    recommendation = "AVOID"
                else:
                    recommendation = "HOLD"
            elif line.startswith("INVESTMENT_SCORE:"):
                try:
                    investment_score = float(line.split(":", 1)[1].strip().split()[0])
                except Exception:
                    investment_score = 5.0
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return {
            "recommendation":  recommendation,
            "investment_score": investment_score,
            "reasoning":       reasoning,
            "current_step":    "advisory_generator",
        }
    except Exception as e:
        return {"error": f"Advisory generation failed: {e}", "current_step": "advisory_generator"}


# Node 5: Report Formatter 

def report_formatter_node(state: AgentState) -> Dict[str, Any]:
    """
    Assemble all node outputs into a structured final report dict
    with clearly labelled sections for the Streamlit UI to render.
    """
    try:
        predicted = state.get("predicted_price", 0)
        low = state.get("price_range_low", 0)
        high = state.get("price_range_high", 0)
        confidence = state.get("price_confidence", "Medium")
        market_summary = state.get("market_summary", "—")
        comps_summary = state.get("comps_summary", "—")
        recommendation = state.get("recommendation", "HOLD")
        investment_score = state.get("investment_score", 5.0)
        reasoning = state.get("reasoning", "—")
        comps = state.get("comparables", [])

        # Format comparables table text
        if comps:
            comps_table = "\n".join([
                f"• ${c['sale_price']:,.0f}  |  {c['gr_liv_area']:,.0f} sqft  |  "
                f"Qual {c['overall_qual']}/10  |  Built {c['year_built']}  |  "
                f"Match: {c['similarity_score']:.0%}"
                for c in comps
            ])
        else:
            comps_table = "No comparable properties found."

        final_report = {
            "valuation_summary": (
                f"Estimated Market Value: **${predicted:,.0f}**\n"
                f"Price Range: ${low:,.0f} – ${high:,.0f}\n"
                f"Model Confidence: {confidence}"
            ),
            "market_view": market_summary,
            "comparable_sales": comps_table,
            "recommendation": recommendation,
            "investment_score": str(round(investment_score, 1)),
            "advisory_reasoning": reasoning,
            "disclaimer": (
                "DISCLAIMER: This report is generated by an AI system and is for "
                "informational purposes only. It does not constitute a licensed real "
                "estate appraisal, financial advice, or a solicitation to buy or sell "
                "property. Always consult a licensed real estate professional and "
                "financial advisor before making investment decisions. Past market "
                "performance does not guarantee future results."
            ),
        }

        return {
            "final_report": final_report,
            "current_step": "report_formatter",
        }
    except Exception as e:
        return {"error": f"Report formatting failed: {e}", "current_step": "report_formatter"}


# Build LangGraph

def build_graph() -> StateGraph:
    """Compile and return the LangGraph agent."""
    builder = StateGraph(AgentState)

    builder.add_node("price_predictor",      price_predictor_node)
    builder.add_node("market_researcher",     market_researcher_node)
    builder.add_node("comparables_analyzer",  comparables_analyzer_node)
    builder.add_node("advisory_generator",    advisory_generator_node)
    builder.add_node("report_formatter",      report_formatter_node)

    # Linear pipeline
    builder.add_edge(START,                  "price_predictor")
    builder.add_edge("price_predictor",      "market_researcher")
    builder.add_edge("market_researcher",    "comparables_analyzer")
    builder.add_edge("comparables_analyzer", "advisory_generator")
    builder.add_edge("advisory_generator",   "report_formatter")
    builder.add_edge("report_formatter",     END)

    return builder.compile()


# Singleton graph instance
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_advisory_agent(
    property_features: dict,
) -> dict:
    """
    Public entry point: run the full agent pipeline.
    Returns the final_report dict (or an error dict).
    """
    graph = get_graph()

    initial_state: AgentState = {
        "property_features":  property_features,
        "predicted_price":    None,
        "price_confidence":   None,
        "price_range_low":    None,
        "price_range_high":   None,
        "market_insights":    None,
        "market_summary":     None,
        "comparables":        None,
        "comps_summary":      None,
        "recommendation":     None,
        "investment_score":   None,
        "reasoning":          None,
        "final_report":       None,
        "error":              None,
        "current_step":       "start",
    }

    result = graph.invoke(initial_state)

    if result.get("error"):
        return {"error": result["error"]}

    return result.get("final_report", {})
