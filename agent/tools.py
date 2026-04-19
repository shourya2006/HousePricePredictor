import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from agent.rag_setup import get_retriever

ROOT_DIR = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / "Model" / "rf_model.jb"
DATASET_PATH = ROOT_DIR / "Dataset" / "AmesHousing.csv"

# ── Model columns────────────────────────────────
MODEL_COLUMNS = [
    'Overall Qual', 'Gr Liv Area', 'Garage Area', '1st Flr SF',
    'Year Built', 'Full Bath', 'Year Remod/Add', 'Mas Vnr Area',
    'Fireplaces', 'BsmtFin SF 1', 'Lot Frontage', 'Wood Deck SF',
    'Open Porch SF', 'Bsmt Full Bath', 'Paved Drive', 'Lot Area',
    'Central Air', 'Roof Style',
]

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(str(MODEL_PATH))
    return _model


# ── Tool 1: Price Predictor ──────────────────────────────────────────────────

def predict_price(features: Dict[str, Any]) -> Dict[str, Any]:
    model = _load_model()

    col_map = {
        'overall_qual':    'Overall Qual',
        'gr_liv_area':     'Gr Liv Area',
        'garage_area':     'Garage Area',
        'first_flr_sf':    '1st Flr SF',
        'year_built':      'Year Built',
        'full_bath':       'Full Bath',
        'year_remod':      'Year Remod/Add',
        'mas_vnr_area':    'Mas Vnr Area',
        'fireplaces':      'Fireplaces',
        'bsmt_fin_sf1':    'BsmtFin SF 1',
        'lot_frontage':    'Lot Frontage',
        'wood_deck_sf':    'Wood Deck SF',
        'open_porch_sf':   'Open Porch SF',
        'bsmt_full_bath':  'Bsmt Full Bath',
        'paved_drive':     'Paved Drive',
        'lot_area':        'Lot Area',
        'central_air':     'Central Air',
        'roof_style':      'Roof Style',
    }

    row = {col_map[k]: v for k, v in features.items() if k in col_map}
    input_df = pd.DataFrame([row], columns=MODEL_COLUMNS)

    predicted = float(model.predict(input_df)[0])
    low = predicted * 0.90
    high = predicted * 1.10

    # Confidence based on overall quality as a proxy
    qual = features.get('overall_qual', 5)
    if qual >= 7:
        confidence = "High"
    elif qual >= 5:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "predicted_price": predicted,
        "price_range_low": low,
        "price_range_high": high,
        "price_confidence": confidence,
    }


# ── Tool 2: Market Insights (RAG) ────────────────────────────────────────────

def retrieve_market_insights(query: str, k: int = 4) -> List[str]:
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


# ── Tool 3: Comparable Properties ────────────────────────────────────────────

def find_comparables(features: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
    df = pd.read_csv(str(DATASET_PATH))

    # Feature weights for similarity (higher = more important)
    comp_features = {
        'Overall Qual': ('overall_qual',    3.0),
        'Gr Liv Area':  ('gr_liv_area',     2.0),
        'Year Built':   ('year_built',      1.5),
        'Garage Area':  ('garage_area',     1.0),
        'Full Bath':    ('full_bath',       1.0),
    }

    required_cols = list(comp_features.keys()) + ['SalePrice']
    df_clean = df[required_cols].dropna().copy()

    # Standardise and compute weighted distances
    distances = np.zeros(len(df_clean))
    for col, (feat_key, weight) in comp_features.items():
        col_std = df_clean[col].std()
        if col_std == 0:
            continue
        user_val = features.get(feat_key, df_clean[col].mean())
        distances += weight * ((df_clean[col] - user_val) / col_std) ** 2

    df_clean['_distance'] = np.sqrt(distances)
    df_clean['_similarity'] = 1 / (1 + df_clean['_distance'])

    top = df_clean.nsmallest(n, '_distance')

    comps = []
    for _, row in top.iterrows():
        comps.append({
            "sale_price":       round(float(row['SalePrice']), 2),
            "gr_liv_area":      round(float(row['Gr Liv Area']), 1),
            "overall_qual":     int(row['Overall Qual']),
            "year_built":       int(row['Year Built']),
            "garage_area":      round(float(row['Garage Area']), 1),
            "full_bath":        int(row['Full Bath']),
            "similarity_score": round(float(row['_similarity']), 3),
        })
    return comps
