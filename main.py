import os
import pandas as pd
import joblib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Real Estate Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 0; }

/* ── App Header ── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2d4a 100%);
    border-radius: 14px;
    padding: 24px 36px;
    margin-bottom: 20px;
    border: 1px solid rgba(99,179,237,0.2);
}
.app-title   { font-size: 32px; font-weight: 700; color: #e2e8f0; margin: 0 0 4px 0; letter-spacing: -0.5px; }
.app-title span { color: #63b3ed; }
.app-subtitle { font-size: 13px; color: #64748b; margin: 0; }

/* ── Section headers ── */
.sec-hdr {
    color: #94a3b8; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.4px;
    padding-bottom: 6px; border-bottom: 1px solid #1e293b; margin-bottom: 14px;
}

/* ── Left panel ── */
.left-panel {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 14px; padding: 20px 18px; height: 100%;
}

/* ── Chat container ── */
.chat-wrapper {
    background: #0a0f1e;
    border: 1px solid #1e293b;
    border-radius: 14px;
    overflow: hidden;
}
.chat-topbar {
    background: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.chat-topbar-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; display: inline-block; }
.chat-topbar-title { color: #cbd5e1; font-size: 13px; font-weight: 600; }
.chat-topbar-sub   { color: #475569; font-size: 11px; margin-left: auto; }

/* ── Tool badges ── */
.tool-badge {
    display: inline-block; border-radius: 99px;
    padding: 2px 10px; font-size: 11px; font-weight: 600;
    margin: 2px 3px 0 0; border: 1px solid rgba(255,255,255,0.1);
}
.badge-price  { background: rgba(29,78,216,0.25);  color: #93c5fd; }
.badge-rag    { background: rgba(5,150,105,0.25);  color: #6ee7b7; }
.badge-comps  { background: rgba(124,58,237,0.25); color: #c4b5fd; }

/* ── Price card in Tab 1 ── */
.price-card {
    background: linear-gradient(135deg, #1e3a5f, #0f2d4a);
    border: 1px solid rgba(99,179,237,0.35); border-radius: 14px;
    padding: 28px 32px; text-align: center; margin-top: 16px;
}
.price-value { color: #63b3ed; font-size: 40px; font-weight: 700; }
.price-range { color: #64748b; font-size: 13px; margin-top: 4px; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500; color: #64748b; padding: 8px 22px; }
.stTabs [aria-selected="true"] { color: #63b3ed !important; border-bottom-color: #63b3ed !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; font-size: 14px; padding: 10px 0;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(29,78,216,0.35);
}

/* ── Generate PDF button ── */
.pdf-btn > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
}
.pdf-btn > button:hover {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.35) !important;
}

/* ── Clear chat ── */
.clear-btn > button {
    background: #1e293b !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    padding: 6px 0 !important;
}

/* User chat message */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: #1d4ed8 !important;
    border-radius: 14px 14px 2px 14px !important;
    margin-left: auto;
    margin-right: 0;
}

/* Assistant chat message */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px 14px 14px 2px !important;
}

/* ── Divider ── */
.styled-hr { border: none; border-top: 1px solid #1e293b; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


MODEL_COLUMNS = [
    'Overall Qual', 'Gr Liv Area', 'Garage Area', '1st Flr SF',
    'Year Built', 'Full Bath', 'Year Remod/Add', 'Mas Vnr Area',
    'Fireplaces', 'BsmtFin SF 1', 'Lot Frontage', 'Wood Deck SF',
    'Open Porch SF', 'Bsmt Full Bath', 'Paved Drive', 'Lot Area',
    'Central Air', 'Roof Style',
]
paved_drive_map = {'No Paved Drive (N)': 0, 'Partial Paved (P)': 1, 'Paved (Y)': 2}
central_air_map = {'No (N)': 0, 'Yes (Y)': 1}
roof_style_map  = {'Flat': 0, 'Gable': 1, 'Gambrel': 2, 'Hip': 3, 'Mansard': 4, 'Shed': 5}
horizon_map = {
    'Short-term (< 2 yrs)':    'short-term',
    'Medium-term (2–5 yrs)':   'medium-term',
    'Long-term (5–10 yrs)':    'long-term',
    'Very long-term (10+ yrs)':'very long-term',
}
strategy_map = {
    'Buy & Hold':        'buy and hold',
    'Fix & Flip':        'fix and flip',
    'Primary Residence': 'primary residence',
    'House Hacking':     'house hacking',
}

TOOL_BADGE_HTML = {
    "predict_property_price":     '<span class="tool-badge badge-price">🔢 Price Predictor</span>',
    "get_market_insights":        '<span class="tool-badge badge-rag">🔍 Market RAG</span>',
    "find_comparable_properties": '<span class="tool-badge badge-comps">📊 Comparable Sales</span>',
}


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "generating_pdf" not in st.session_state:
    st.session_state.generating_pdf = False


@st.cache_resource
def load_model():
    return joblib.load('./Model/rf_model.jb')

model = load_model()


st.markdown("""
<div class="app-header">
    <p class="app-title">🏡 AI Real Estate <span>Advisor</span></p>
    <p class="app-subtitle">
        LangGraph Agentic AI · ChromaDB RAG · OpenAI GPT-4o-mini · Ames Housing Dataset
    </p>
</div>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["🔢  Price Estimator", "🤖  AI Advisor"])



with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    inp = {}

    with c1:
        st.markdown('<div class="sec-hdr">Property Details</div>', unsafe_allow_html=True)
        inp['Overall Qual']  = st.slider('Overall Quality (1–10)', 1, 10, 5, key='t1_q')
        inp['Gr Liv Area']   = st.number_input('Living Area (sq ft)', 0.0, value=1500.0, step=50.0, key='t1_l')
        inp['1st Flr SF']    = st.number_input('First Floor (sq ft)', 0.0, value=1000.0, step=50.0, key='t1_f')
        inp['Lot Frontage']  = st.number_input('Street Connected (ft)', 0.0, value=65.0, step=5.0, key='t1_lf')
        inp['Lot Area']      = st.number_input('Lot Size (sq ft)', 0.0, value=8000.0, step=100.0, key='t1_la')
        sr = st.selectbox('Roof Style', list(roof_style_map), index=1, key='t1_r')
        inp['Roof Style'] = roof_style_map[sr]

    with c2:
        st.markdown('<div class="sec-hdr">Construction & Age</div>', unsafe_allow_html=True)
        inp['Year Built']     = st.number_input('Year Built', 1800, 2024, 1980, step=1, key='t1_yb')
        inp['Year Remod/Add'] = st.number_input('Remodel Year', 1800, 2024, 2000, step=1, key='t1_yr')
        inp['Mas Vnr Area']   = st.number_input('Masonry Veneer (sq ft)', 0.0, value=0.0, step=10.0, key='t1_mv')
        sa = st.selectbox('Central Air', list(central_air_map), index=1, key='t1_ca')
        inp['Central Air'] = central_air_map[sa]
        sd = st.selectbox('Paved Driveway', list(paved_drive_map), index=2, key='t1_pd')
        inp['Paved Drive'] = paved_drive_map[sd]
        inp['Fireplaces']     = st.slider('Fireplaces', 0, 5, 0, key='t1_fp')

    with c3:
        st.markdown('<div class="sec-hdr">Amenities & Basement</div>', unsafe_allow_html=True)
        inp['Full Bath']      = st.slider('Full Bathrooms', 0, 5, 2, key='t1_fb')
        inp['Garage Area']    = st.number_input('Garage (sq ft)', 0.0, value=400.0, step=50.0, key='t1_g')
        inp['Bsmt Full Bath'] = st.slider('Basement Full Baths', 0, 3, 0, key='t1_bb')
        inp['BsmtFin SF 1']   = st.number_input('Finished Basement (sq ft)', 0.0, value=500.0, step=50.0, key='t1_bs')
        inp['Wood Deck SF']   = st.number_input('Wood Deck (sq ft)', 0.0, value=0.0, step=10.0, key='t1_wd')
        inp['Open Porch SF']  = st.number_input('Open Porch (sq ft)', 0.0, value=0.0, step=10.0, key='t1_op')

    st.markdown('<hr class="styled-hr">', unsafe_allow_html=True)
    _, cc, _ = st.columns([1, 1, 1])
    with cc:
        if st.button("Generate Price Estimate", use_container_width=True, key='t1_btn'):
            try:
                df   = pd.DataFrame([inp], columns=MODEL_COLUMNS)
                pred = model.predict(df)[0]
                lo, hi = pred * 0.90, pred * 1.10
                st.markdown(f"""
                <div class="price-card">
                    <p style="color:#94a3b8;font-size:12px;margin:0 0 4px 0;">Estimated Market Value</p>
                    <p class="price-value">${pred:,.0f}</p>
                    <p class="price-range">Confidence Range: ${lo:,.0f} – ${hi:,.0f}</p>
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")


with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    # Split into left panel (property context) + right panel (chat)
    left_col, right_col = st.columns([1, 2.6], gap="large")

    # ── LEFT PANEL: Property Context ─────────────────────────────────────────
    with left_col:
        st.markdown('<div class="sec-hdr">🏠 Property Context</div>', unsafe_allow_html=True)
        st.caption("Scroll to edit property details and preferences.")

        # ── Scrollable inputs panel ────────────────────────────────────────
        with st.container(height=400):
            st.markdown("**Property Features**")
            ctx_qual    = st.slider("Overall Quality (1–10)", 1, 10, 6, key="ctx_q")
            ctx_area    = st.number_input("Living Area (sq ft)", 0.0, value=1500.0, step=50.0, key="ctx_la")
            ctx_yr      = st.number_input("Year Built", 1800, 2024, 1990, step=1, key="ctx_yr")
            ctx_rem     = st.number_input("Remodel Year", 1800, 2024, 2005, step=1, key="ctx_rem")
            ctx_garage  = st.number_input("Garage Area (sq ft)", 0.0, value=400.0, step=50.0, key="ctx_g")
            ctx_bath    = st.slider("Full Bathrooms", 0, 5, 2, key="ctx_b")
            ctx_fire    = st.slider("Fireplaces", 0, 5, 0, key="ctx_fp")
            ctx_lot     = st.number_input("Lot Area (sq ft)", 0.0, value=8000.0, step=100.0, key="ctx_lot")
            ctx_bsmt    = st.number_input("Finished Basement (sq ft)", 0.0, value=0.0, step=50.0, key="ctx_bsmt")
            ctx_front   = st.number_input("Lot Frontage (ft)", 0.0, value=65.0, step=5.0, key="ctx_front")
            ctx_1stflr  = st.number_input("First Floor (sq ft)", 0.0, value=1000.0, step=50.0, key="ctx_1f")
            ctx_air_lbl = st.selectbox("Central Air", list(central_air_map), index=1, key="ctx_air")
            ctx_drv_lbl = st.selectbox("Paved Driveway", list(paved_drive_map), index=2, key="ctx_drv")
            ctx_roof_lbl= st.selectbox("Roof Style", list(roof_style_map), index=1, key="ctx_roof")



        property_context = {
            "overall_qual":   ctx_qual,
            "gr_liv_area":    ctx_area,
            "year_built":     int(ctx_yr),
            "year_remod":     int(ctx_rem),
            "garage_area":    ctx_garage,
            "full_bath":      ctx_bath,
            "fireplaces":     ctx_fire,
            "lot_area":       ctx_lot,
            "bsmt_fin_sf1":   ctx_bsmt,
            "lot_frontage":   ctx_front,
            "first_flr_sf":   ctx_1stflr,
            "central_air":    central_air_map[ctx_air_lbl],
            "paved_drive":    paved_drive_map[ctx_drv_lbl],
            "roof_style":     roof_style_map[ctx_roof_lbl],
            "mas_vnr_area":   0.0,
            "wood_deck_sf":   0.0,
            "open_porch_sf":  0.0,
            "bsmt_full_bath": 0,
        }


        st.markdown('<hr class="styled-hr">', unsafe_allow_html=True)

        # ── PDF Report Generation ────────
        st.markdown('<div class="sec-hdr">📄 Full Advisory Report</div>', unsafe_allow_html=True)
        st.caption("Runs all 5 agent nodes and generates a downloadable PDF.")

        gen_pdf_btn = st.button(
            "Generate PDF Report",
            use_container_width=True,
            key="gen_pdf_btn",
        )

        if gen_pdf_btn:
            with st.spinner("Running full advisory pipeline…"):
                try:
                    from agent.graph import run_advisory_agent
                    from agent.pdf_generator import generate_pdf_report

                    report_data = run_advisory_agent(
                        property_features=property_context,
                    )

                    if "error" not in report_data:
                        pdf_bytes = generate_pdf_report(
                            report_data=report_data,
                            property_features=property_context,
                        )
                        st.session_state.pdf_bytes = pdf_bytes
                        st.session_state.pdf_ready = True
                    else:
                        st.error(f"Agent error: {report_data['error']}")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.pdf_ready and st.session_state.pdf_bytes:
            st.success("Report ready!")
            st.download_button(
                label="Download PDF",
                data=st.session_state.pdf_bytes,
                file_name="real_estate_advisory_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_btn",
            )

        st.markdown('<hr class="styled-hr">', unsafe_allow_html=True)


        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("Clear Chat History", use_container_width=True, key="clear_btn"):
            st.session_state.chat_messages = []
            st.session_state.pdf_bytes = None
            st.session_state.pdf_ready = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT PANEL: Chat Interface ───────────────────────────────────────────
    with right_col:


        st.markdown("""
        <div class="chat-topbar" style="background:#0f172a; border:1px solid #1e293b;
             border-radius:14px 14px 0 0; padding:12px 20px; display:flex; align-items:center; gap:10px;">
            <span class="chat-topbar-dot"></span>
            <span class="chat-topbar-title">AI Real Estate Advisor</span>
            <span class="chat-topbar-sub">Auto tool-routing · GPT-4o-mini · RAG</span>
        </div>
        """, unsafe_allow_html=True)


        if not st.session_state.chat_messages:
            st.markdown("""
            <div style="background:#0a0f1e; border:1px solid #1e293b; border-top:none;
                 border-radius:0; padding:20px 24px 0 24px;">
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:12px;
                     padding:20px 22px; margin-bottom: 8px;">
                    <p style="color:#63b3ed; font-size:14px; font-weight:600; margin:0 0 8px 0;">
                        👋 Hello! I'm your AI Real Estate Advisor.
                    </p>
                    <p style="color:#94a3b8; font-size:13px; line-height:1.7; margin:0;">
                        I automatically pick the right tool based on your question. Try asking me:
                    </p>
                    <ul style="color:#64748b; font-size:13px; margin-top:8px; line-height:2;">
                        <li>💰 <em>"What's the estimated price for this property?"</em></li>
                        <li>📈 <em>"Is this a good investment? What are the market trends?"</em></li>
                        <li>🏘️ <em>"Show me comparable properties that have sold recently."</em></li>
                        <li>🧠 <em>"Analyze the risk factors for this purchase."</em></li>
                    </ul>
                    <p style="color:#475569; font-size:11px; margin: 8px 0 0 0;">
                        To generate a full PDF advisory report, click <strong>"📄 Generate PDF Report"</strong> on the left.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Message History ───────────────────────────────────────────────────
        chat_container = st.container(height=520)
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"].replace("$", "&#36;"))
                else:
                    with st.chat_message("assistant"):

                        tools_used = msg.get("tools_used", [])
                        if tools_used:
                            badges = "".join(
                                TOOL_BADGE_HTML.get(t, f'<span class="tool-badge">{t}</span>')
                                for t in tools_used
                            )
                            st.markdown(
                                f'<div style="margin-bottom:6px;">{badges}</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown(msg["content"].replace("$", "&#36;"))

        # ── Chat Input ────────────────────────────────────────────────────────
        user_input = st.chat_input(
            "Ask me about price, market trends, comparable sales, investment risk…",
            key="chat_input",
        )

        if user_input:
            from agent.chat_agent import stream_chat


            st.session_state.chat_messages.append({
                "role":    "user",
                "content": user_input,
            })


            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)


                with st.chat_message("assistant"):
                    tools_used: list = []


                    badge_placeholder = st.empty()

                    def _stream_gen():
                        """Strips tool events from the generator, updates badges live."""
                        for event in stream_chat(
                            user_message=user_input,
                            conversation_history=st.session_state.chat_messages[:-1],
                            property_context=property_context,
                        ):
                            if isinstance(event, dict) and event.get("type") == "tool":
                                tools_used.append(event["name"])
                                badges_html = "".join(
                                    TOOL_BADGE_HTML.get(t, f'<span class="tool-badge">{t}</span>')
                                    for t in tools_used
                                )
                                badge_placeholder.markdown(
                                    f'<div style="margin-bottom:6px;">{badges_html}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                yield str(event)


                    response_text = ""
                    stream_md = st.empty()
                    try:
                        for chunk in _stream_gen():
                            response_text += chunk

                            safe_text = response_text.replace("$", "&#36;")

                            stream_md.markdown(safe_text + "▌")

                        stream_md.markdown(response_text.replace("$", "&#36;"))
                    except Exception as e:
                        import traceback
                        response_text = f"⚠️ Error: {e}\n```\n{traceback.format_exc()}\n```"
                        stream_md.markdown(response_text)

            st.session_state.chat_messages.append({
                "role":       "assistant",
                "content":    response_text,
                "tools_used": list(tools_used),
            })
