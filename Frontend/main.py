import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="House Price Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 40px;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 5px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        color: #E2E8F0;
        margin-bottom: 40px;
    }
    .section-header {
        color: #F8FAFC;
        font-weight: 500;
        padding-bottom: 10px;
        border-bottom: 1px solid #94A3B8;
        margin-bottom: 20px;
    }
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
        border: none;
        border-top: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

def load_model():
    return joblib.load('rf_model.jb')

model = load_model()

st.markdown('<div class="main-header">House Price Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter the property details below to generate an estimated market value.</div>', unsafe_allow_html=True)

expected_columns = [
    'Overall Qual', 'Gr Liv Area', 'Garage Area', '1st Flr SF',
    'Year Built', 'Full Bath', 'Year Remod/Add', 'Mas Vnr Area',
    'Fireplaces', 'BsmtFin SF 1', 'Lot Frontage', 'Wood Deck SF',
    'Open Porch SF', 'Bsmt Full Bath', 'Paved Drive', 'Lot Area',
    'Central Air', 'Roof Style'
]


paved_drive_map = {'No Paved Drive (N)': 0, 'Partial Paved (P)': 1, 'Paved (Y)': 2}
central_air_map = {'No (N)': 0, 'Yes (Y)': 1}
roof_style_map = {
    'Flat': 0, 'Gable': 1, 'Gambrel': 2, 
    'Hip': 3, 'Mansard': 4, 'Shed': 5
}

col1, col2, col3 = st.columns(3, gap="large")

input_data = {}

with col1:
    st.markdown('<div class="section-header">Property Details</div>', unsafe_allow_html=True)
    input_data['Overall Qual'] = st.slider('Overall Quality (1-10)', min_value=1, max_value=10, value=5)
    input_data['Gr Liv Area'] = st.number_input('Above Grade Living Area (sq ft)', min_value=0.0, value=1500.0, step=50.0)
    input_data['1st Flr SF'] = st.number_input('First Floor Area (sq ft)', min_value=0.0, value=1000.0, step=50.0)
    input_data['Lot Frontage'] = st.number_input('Street Connected Length (ft)', min_value=0.0, value=65.0, step=10.0)
    input_data['Lot Area'] = st.number_input('Total Lot Size (sq ft)', min_value=0.0, value=8000.0, step=100.0)
    
    selected_roof = st.selectbox('Roof Style', list(roof_style_map.keys()), index=1)
    input_data['Roof Style'] = roof_style_map[selected_roof]


with col2:
    st.markdown('<div class="section-header">Construction & Age</div>', unsafe_allow_html=True)
    input_data['Year Built'] = st.number_input('Original Construction Year', min_value=1800, max_value=2024, value=1980, step=1)
    input_data['Year Remod/Add'] = st.number_input('Remodel Year', min_value=1800, max_value=2024, value=2000, step=1)
    input_data['Mas Vnr Area'] = st.number_input('Masonry Veneer Area (sq ft)', min_value=0.0, value=0.0, step=10.0)
    
    selected_air = st.selectbox('Central Air Conditioning', list(central_air_map.keys()), index=1)
    input_data['Central Air'] = central_air_map[selected_air]
    
    selected_drive = st.selectbox('Paved Driveway', list(paved_drive_map.keys()), index=2)
    input_data['Paved Drive'] = paved_drive_map[selected_drive]
    
    input_data['Fireplaces'] = st.slider('Number of Fireplaces', min_value=0, max_value=5, value=0)


with col3:
    st.markdown('<div class="section-header">Amenities & Basement</div>', unsafe_allow_html=True)
    input_data['Full Bath'] = st.slider('Full Bathrooms Above Grade', min_value=0, max_value=5, value=2)
    input_data['Garage Area'] = st.number_input('Garage Area (sq ft)', min_value=0.0, value=400.0, step=50.0)
    
    input_data['Bsmt Full Bath'] = st.slider('Basement Full Bathrooms', min_value=0.0, max_value=3.0, value=0.0, step=1.0)
    input_data['BsmtFin SF 1'] = st.number_input('Finished Basement Area (sq ft)', min_value=0.0, value=500.0, step=50.0)
    
    input_data['Wood Deck SF'] = st.number_input('Wood Deck Area (sq ft)', min_value=0.0, value=0.0, step=10.0)
    input_data['Open Porch SF'] = st.number_input('Open Porch Area (sq ft)', min_value=0.0, value=0.0, step=10.0)


st.markdown('<hr>', unsafe_allow_html=True)

_, center_col, _ = st.columns([1, 1, 1])

with center_col:
    predict_btn = st.button("Generate Estimate", use_container_width=True, type="primary")

if predict_btn:
    try:
        input_df = pd.DataFrame([input_data], columns=expected_columns)
        with st.spinner('Calculating valuation...'):
            prediction = model.predict(input_df)[0]
        
        st.success(f'Estimated Property Value: **${prediction:,.2f}**')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
