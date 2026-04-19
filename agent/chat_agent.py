import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
)
from agent.tools import predict_price, retrieve_market_insights, find_comparables


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI real estate advisor with access to three powerful tools:

1. **predict_property_price** — Uses a trained Random Forest ML model to estimate property market value.
   → Use when asked about: price, value, worth, valuation, how much a house costs, estimate.

2. **get_market_insights** — RAG retrieval from a real estate knowledge base.
   → Use when asked about: market trends, investment advice, risk factors, appreciation,
     rental yield, market conditions, should I buy, is it a good investment.

3. **find_comparable_properties** — Finds similar sold properties from the Ames Housing dataset.
   → Use when asked about: comparable sales, similar homes, comps, what similar homes sold for.

GUIDELINES:
- Always use the appropriate tool(s) — never guess or make up numbers.
- For comprehensive questions, use multiple tools and synthesise the results.
- Format all dollar amounts with commas and $ signs.
- Keep responses clear, professional, and advice-oriented.
- When the user asks to "generate a report" or "create a PDF", tell them to click
  the "Generate PDF Report" button in the panel on the left — you cannot generate
  the PDF yourself, but clicking the button will run the full advisory pipeline.
- If the user hasn't specified property details, use the DEFAULT PROPERTY CONTEXT provided.

{context_block}"""


def _build_system_prompt(property_context: dict) -> str:
    if not property_context:
        return SYSTEM_PROMPT.replace("{context_block}", "")

    roof_names = {0: 'Flat', 1: 'Gable', 2: 'Gambrel', 3: 'Hip', 4: 'Mansard', 5: 'Shed'}
    ctx = f"""DEFAULT PROPERTY CONTEXT (use these values if the user hasn't specified):
- Overall Quality: {property_context.get('overall_qual', 6)}/10
- Living Area: {property_context.get('gr_liv_area', 1500):,.0f} sq ft
- First Floor: {property_context.get('first_flr_sf', 1000):,.0f} sq ft
- Lot Area: {property_context.get('lot_area', 8000):,.0f} sq ft
- Year Built: {property_context.get('year_built', 1990)}
- Remodel Year: {property_context.get('year_remod', 2005)}
- Full Bathrooms: {property_context.get('full_bath', 2)}
- Fireplaces: {property_context.get('fireplaces', 0)}
- Garage Area: {property_context.get('garage_area', 400):,.0f} sq ft
- Finished Basement: {property_context.get('bsmt_fin_sf1', 0):,.0f} sq ft
- Central Air: {"Yes" if property_context.get('central_air', 1) == 1 else "No"}
- Paved Driveway: {"Yes" if property_context.get('paved_drive', 2) == 2 else "Partial/No"}
- Roof Style: {roof_names.get(property_context.get('roof_style', 1), 'Gable')}"""

    return SYSTEM_PROMPT.replace("{context_block}", ctx)


@tool
def predict_property_price(
    overall_qual: int = 6,
    gr_liv_area: float = 1500.0,
    year_built: int = 1990,
    garage_area: float = 400.0,
    full_bath: int = 2,
    year_remod: int = 2005,
    fireplaces: int = 0,
    bsmt_fin_sf1: float = 0.0,
    lot_area: float = 8000.0,
    lot_frontage: float = 65.0,
    wood_deck_sf: float = 0.0,
    open_porch_sf: float = 0.0,
    bsmt_full_bath: int = 0,
    paved_drive: int = 2,
    central_air: int = 1,
    roof_style: int = 1,
    mas_vnr_area: float = 0.0,
    first_flr_sf: float = 1000.0,
) -> str:
    """Predicts a property's market price using a Random Forest model."""
    features = {
        'overall_qual':   overall_qual,
        'gr_liv_area':    gr_liv_area,
        'garage_area':    garage_area,
        'first_flr_sf':   first_flr_sf,
        'year_built':     year_built,
        'full_bath':      full_bath,
        'year_remod':     year_remod,
        'mas_vnr_area':   mas_vnr_area,
        'fireplaces':     fireplaces,
        'bsmt_fin_sf1':   bsmt_fin_sf1,
        'lot_frontage':   lot_frontage,
        'wood_deck_sf':   wood_deck_sf,
        'open_porch_sf':  open_porch_sf,
        'bsmt_full_bath': bsmt_full_bath,
        'paved_drive':    paved_drive,
        'lot_area':       lot_area,
        'central_air':    central_air,
        'roof_style':     roof_style,
    }
    result = predict_price(features)
    p = result['predicted_price']
    return (
        f"Predicted Market Value: ${p:,.0f}\n"
        f"Price Range: ${result['price_range_low']:,.0f} – ${result['price_range_high']:,.0f}\n"
        f"Model Confidence: {result['price_confidence']}\n"
        f"Key factors: Quality {overall_qual}/10, {gr_liv_area:,.0f} sqft living area, "
        f"built {year_built}, {garage_area:,.0f} sqft garage."
    )


@tool
def get_market_insights(query: str) -> str:
    """Retrieve market trends and risk analysis from the knowledge base."""
    chunks = retrieve_market_insights(query, k=5)
    return "\n\n---\n\n".join(chunks)


@tool
def find_comparable_properties(
    overall_qual: int = 6,
    gr_liv_area: float = 1500.0,
    year_built: int = 1990,
    garage_area: float = 400.0,
    full_bath: int = 2,
    n: int = 5,
) -> str:
    """Find comparable sold properties based on input features."""
    features = {
        'overall_qual': overall_qual,
        'gr_liv_area':  gr_liv_area,
        'year_built':   year_built,
        'garage_area':  garage_area,
        'full_bath':    full_bath,
    }
    comps = find_comparables(features, n=n)
    if not comps:
        return "No comparable properties found in the dataset."

    lines = [f"Top {len(comps)} comparable properties:\n"]
    for i, c in enumerate(comps, 1):
        lines.append(
            f"{i}. Sale Price: ${c['sale_price']:,.0f} | "
            f"Area: {c['gr_liv_area']:,.0f} sqft | "
            f"Quality: {c['overall_qual']}/10 | "
            f"Built: {c['year_built']} | "
            f"Garage: {c['garage_area']:,.0f} sqft | "
            f"Match: {c['similarity_score']:.0%}"
        )
    avg = sum(c['sale_price'] for c in comps) / len(comps)
    lines.append(f"\nAverage comparable sale price: ${avg:,.0f}")
    return "\n".join(lines)


# ── Tool Registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [predict_property_price, get_market_insights, find_comparable_properties]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

TOOL_DISPLAY = {
    "predict_property_price":    ("🔢", "Price Predictor",     "#1d4ed8"),
    "get_market_insights":       ("🔍", "Market Insights RAG", "#059669"),
    "find_comparable_properties":("📊", "Comparable Sales",    "#7c3aed"),
}


# ── LLM ──────────────────────────────────────────────────────────────────────

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.35,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm


# ── Core Chat Function ────────────────────────────────────────────────────────

def run_chat(
    user_message: str,
    conversation_history: list,
    property_context: dict = None,
    max_tool_rounds: int = 6,
) -> dict:
    llm_with_tools = _get_llm().bind_tools(ALL_TOOLS)

    system_prompt = _build_system_prompt(property_context)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    tools_used = []

    for _ in range(max_tool_rounds):
        response = llm_with_tools.invoke(messages)
        if not response.tool_calls:
            return {
                "response":   response.content,
                "tools_used": tools_used,
            }

        messages.append(response)

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tools_used.append(tool_name)

            tool_obj = TOOL_MAP.get(tool_name)
            if tool_obj is None:
                tool_result = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    tool_result = tool_obj.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Tool error: {e}"

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                    name=tool_name,
                )
            )
    final = llm_with_tools.invoke(messages)
    return {
        "response":   final.content or "I wasn't able to complete that request. Please try again.",
        "tools_used": tools_used,
    }


# ── Streaming Chat Function ───────────────────────────────────────────────────

def stream_chat(
    user_message: str,
    conversation_history: list,
    property_context: dict = None,
    max_tool_rounds: int = 6,
):
    llm_with_tools = _get_llm().bind_tools(ALL_TOOLS)
    system_prompt = _build_system_prompt(property_context)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    for _ in range(max_tool_rounds):
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            break

        messages.append(response)

        for tc in response.tool_calls:
            yield {"type": "tool", "name": tc["name"]}

            tool_obj = TOOL_MAP.get(tc["name"])
            try:
                tool_result = tool_obj.invoke(tc["args"]) if tool_obj else f"Unknown tool: {tc['name']}"
            except Exception as e:
                tool_result = f"Tool error: {e}"

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )

    try:
        for chunk in _get_llm().stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"\n\nStream error: {e}"
