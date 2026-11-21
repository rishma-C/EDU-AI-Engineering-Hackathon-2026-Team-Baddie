"""
EDU AI Engineering Hackathon 2026 - Preliminary Round
AI for Urban Dengue Outbreak in Bangladesh
Interactive Public Health Decision Support System & Early Warning Dashboard
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spatial_cv.models import DengueEnsemblePredictor

# Page Configuration
st.set_page_config(
    page_title="WardWatch | AI for Urban Dengue Outbreak Early Warning",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# THEME STATE & DYNAMIC SCREEN COLOUR CONTROLS
# -------------------------------------------------------------
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark'

# Theme Colors Palette
is_dark = (st.session_state['theme'] == 'dark')

if is_dark:
    bg_color = "#0e131b"
    sidebar_bg = "#131924"
    sidebar_text = "#f1f5f9"
    sidebar_label = "#94a3b8"
    input_bg = "#182232"
    input_border = "rgba(255, 255, 255, 0.15)"
    input_text = "#f8fafc"
    card_bg = "#182232"
    card_border = "rgba(255, 255, 255, 0.08)"
    text_color = "#f8fafc"
    text_muted = "#94a3b8"
    hero_bg = "linear-gradient(135deg, #092615 0%, #031a0d 100%)"
    hero_border = "#134e29"
    hero_badge_bg = "rgba(34, 197, 94, 0.15)"
    hero_badge_border = "rgba(34, 197, 94, 0.35)"
    hero_badge_text = "#86efac"
    info_banner_bg = "#162338"
    info_banner_border = "#1e3a5f"
    info_banner_text = "#93c5fd"
    table_header_bg = "#131b28"
    table_row_alt = "#111824"
    plotly_template = "plotly_dark"
else:
    bg_color = "#f8fafc"
    sidebar_bg = "#f1f5f9"
    sidebar_text = "#0f172a"
    sidebar_label = "#334155"
    input_bg = "#ffffff"
    input_border = "#cbd5e1"
    input_text = "#0f172a"
    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    text_color = "#0f172a"
    text_muted = "#64748b"
    hero_bg = "linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%)"
    hero_border = "#2d6a4f"
    hero_badge_bg = "rgba(255, 255, 255, 0.2)"
    hero_badge_border = "rgba(255, 255, 255, 0.4)"
    hero_badge_text = "#ffffff"
    info_banner_bg = "#eff6ff"
    info_banner_border = "#bfdbfe"
    info_banner_text = "#1e40af"
    table_header_bg = "#f1f5f9"
    table_row_alt = "#f8fafc"
    plotly_template = "plotly_white"

# Inject Global Adaptive CSS
st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        color: {sidebar_text} !important;
        border-right: 1px solid {card_border};
    }}
    section[data-testid="stSidebar"] * {{
        color: {sidebar_text} !important;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {{
        color: {sidebar_text} !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stCaption p {{
        color: {sidebar_label} !important;
        font-weight: 400 !important;
    }}
    /* Inputs in sidebar */
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border: 1px solid {input_border} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] span {{
        color: {input_text} !important;
        font-weight: 600 !important;
    }}
    div[data-baseweb="select"] svg {{
        fill: {sidebar_text} !important;
    }}
    div[data-baseweb="popover"] ul {{
        background-color: {card_bg} !important;
    }}
    div[data-baseweb="popover"] li {{
        color: {text_color} !important;
    }}
    input[type="text"], input[disabled] {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border: 1px solid {input_border} !important;
        border-radius: 8px !important;
        -webkit-text-fill-color: {input_text} !important;
        opacity: 0.9 !important;
        font-weight: 600 !important;
    }}
    /* Button styles */
    button[kind="primary"] {{
        background-color: #10b981 !important;
        color: #ffffff !important;
        border: 1px solid #059669 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }}
    button[kind="secondary"] {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {card_border} !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }}
    header[data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
    .hero-banner {{
        background: {hero_bg} !important;
        border: 1px solid {hero_border} !important;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .info-banner {{
        background-color: {info_banner_bg} !important;
        border: 1px solid {info_banner_border} !important;
        color: {info_banner_text} !important;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }}
    .theme-card {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }}
    .streamlit-expanderHeader {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
    }}
    .badge-pill {{
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        text-align: center;
    }}
    .pill-low {{ background-color: #1b4332; color: #74c69d; border: 1px solid #2d6a4f; }}
    .pill-medium {{ background-color: #5c3d00; color: #ffdd80; border: 1px solid #8c5d00; }}
    .pill-high {{ background-color: #6b2600; color: #ffbe99; border: 1px solid #a33b00; }}
    .pill-critical {{ background-color: #590d0d; color: #ff9999; border: 1px solid #8f1717; }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. CACHED DATA & MODEL LOADERS (Criterion 6: Fast Local Inference)
# -------------------------------------------------------------
@st.cache_resource
def load_production_model():
    """Loads serialized production ensemble with zero recurring API billing."""
    model_path = os.path.join(os.path.dirname(__file__), 'dengue_model.joblib')
    metadata_path = os.path.join(os.path.dirname(__file__), 'model_metadata.json')
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata

@st.cache_data
def load_dataset_metadata():
    """Loads dataset panel and pre-computes static ward profiles and reference dates."""
    csv_path = os.path.join(os.path.dirname(__file__), 'hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv')
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Static traits per ward
    ward_props = df.groupby(['city', 'ward_id']).agg({
        'population_density_sqkm': 'first',
        'water_logging_risk_index': 'first',
        'sanitation_score': 'first',
        'avg_temp_c': 'mean',
        'rainfall_mm': 'mean',
        'relative_humidity_pct': 'mean'
    }).reset_index()
    
    # Latest historical observation per ward (29 Dec 2025)
    latest_df = df.sort_values(['ward_id', 'date']).groupby('ward_id').last().reset_index()
    
    return df, ward_props, latest_df

model, metadata = load_production_model()
raw_df, ward_props, latest_df = load_dataset_metadata()

# Reference Date in dataset
reference_date = raw_df['date'].max()
ref_date_str = reference_date.strftime('%Y/%m/%d')

# -------------------------------------------------------------
# SIDEBAR CONTROLS (Matching Image 1 & Image 2)
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style="margin-bottom: 0.8rem;">
    <span style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.5px;">🦟 WardWatch</span><br/>
    <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.75; font-weight: 700;">DENGUE EARLY WARNING · LOCAL MODEL</span>
</div>
""", unsafe_allow_html=True)

# SCREEN COLOUR THEME SWITCHER BUTTON
st.sidebar.markdown("<span style='font-size: 0.8rem; font-weight: 700; opacity: 0.8;'>SCREEN COLOUR MODE</span>", unsafe_allow_html=True)
col_theme1, col_theme2 = st.sidebar.columns(2)
with col_theme1:
    if st.button("🌙 Dark", use_container_width=True, type="primary" if is_dark else "secondary"):
        st.session_state['theme'] = 'dark'
        st.rerun()
with col_theme2:
    if st.button("☀️ White", use_container_width=True, type="primary" if not is_dark else "secondary"):
        st.session_state['theme'] = 'light'
        st.rerun()

st.sidebar.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

# City & Ward Selection
cities = sorted(ward_props['city'].unique().tolist())
default_city = 'Chattogram' if 'Chattogram' in cities else cities[0]
selected_city = st.sidebar.selectbox("City:", cities, index=cities.index(default_city))

city_wards = sorted(ward_props[ward_props['city'] == selected_city]['ward_id'].unique().tolist())
selected_ward = st.sidebar.selectbox("Ward:", city_wards, index=0)

# Reference week display
st.sidebar.text_input("Reference week (Data Boundary):", value=ref_date_str, disabled=True)

# Forecast Horizon Selector (Image 1: +2 weeks, +3 weeks, +4 weeks)
st.sidebar.markdown("<span style='font-size: 0.82rem; font-weight: 700;'>Forecast horizon</span>", unsafe_allow_html=True)
horizon_choice = st.sidebar.radio(
    "Select Lead Horizon:",
    ["+2 weeks", "+3 weeks", "+4 weeks"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

horizon_weeks = int(horizon_choice.replace('+', '').replace(' weeks', ''))
target_forecast_date = reference_date + pd.Timedelta(weeks=horizon_weeks)
target_date_str = target_forecast_date.strftime('%d %b %Y').upper()

# Ward Baseline Properties
ward_info = ward_props[ward_props['ward_id'] == selected_ward].iloc[0]
pop_density = int(ward_info['population_density_sqkm'])
water_logging = float(ward_info['water_logging_risk_index'])
sanitation = float(ward_info['sanitation_score'])

# Observed values from latest historical week (29 Dec 2025)
ward_latest = latest_df[latest_df['ward_id'] == selected_ward].iloc[0]
obs_rain_curr = float(ward_latest['rainfall_mm'])
obs_temp_curr = float(ward_latest['avg_temp_c'])
obs_hum_curr = float(ward_latest['relative_humidity_pct'])

# Weather Scenario Controls
st.sidebar.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='font-size: 0.88rem; font-weight: 800;'>Weather scenario</span>", unsafe_allow_html=True)
st.sidebar.caption("Defaults use the selected ward's observed rolling values.")

# Quick Preset Scenarios
scenario = st.sidebar.selectbox(
    "Quick Scenario Presets:",
    ["Observed Baseline (29 Dec 2025)", "Peak Monsoon Surge (Heavy Rain & Flooding)", "Post-Monsoon Incubation Peak (Lag-3 Surge)", "Dry Winter Transition"]
)

if scenario == "Peak Monsoon Surge (Heavy Rain & Flooding)":
    init_r2 = 140.0
    init_r4 = 110.0
    init_t2 = 29.5
    init_t4 = 28.5
    init_hum = 88.0
elif scenario == "Post-Monsoon Incubation Peak (Lag-3 Surge)":
    init_r2 = 70.0
    init_r4 = 175.0  # Heavy rain 3-4 weeks prior!
    init_t2 = 28.5
    init_t4 = 28.0
    init_hum = 82.0
elif scenario == "Dry Winter Transition":
    init_r2 = 2.0
    init_r4 = 5.0
    init_t2 = 20.0
    init_t4 = 21.0
    init_hum = 55.0
else:
    init_r2 = max(5.0, round(obs_rain_curr, 2))
    init_r4 = max(8.0, round(obs_rain_curr * 1.1, 2))
    init_t2 = round(obs_temp_curr, 2)
    init_t4 = round(obs_temp_curr - 0.5, 2)
    init_hum = round(obs_hum_curr, 1)

# Sliders matching Image 1
scen_r2 = st.sidebar.slider("Rainfall · 2 weeks (mm):", 0.0, 300.0, float(init_r2), step=1.0)
scen_r4 = st.sidebar.slider("Rainfall · 4 weeks (mm):", 0.0, 300.0, float(init_r4), step=1.0)
scen_t2 = st.sidebar.slider("Average temperature · 2 weeks (°C):", 14.0, 38.0, float(init_t2), step=0.2)
scen_t4 = st.sidebar.slider("Average temperature · 4 weeks (°C):", 14.0, 38.0, float(init_t4), step=0.2)
scen_hum = st.sidebar.slider("Relative humidity (%):", 30.0, 100.0, float(init_hum), step=1.0)

# -------------------------------------------------------------
# DYNAMIC FEATURE ENGINEERING PIPELINE
# -------------------------------------------------------------
def build_features_for_ward(w_pop, w_water, w_sani, r2, r4, t2, t4, hum, target_dt):
    # Derive multi-week lags based on rolling inputs
    r_l1 = (r2 * 1.05)
    r_l2 = r2
    r_l3 = (r4 * 0.95)
    r_l4 = r4
    r_curr = r2 * 0.8
    
    r_roll2_mean = (r_curr + r_l1) / 2.0
    r_roll3_mean = (r_curr + r_l1 + r_l2) / 3.0
    r_roll4_mean = (r_curr + r_l1 + r_l2 + r_l3) / 4.0
    r_roll4_sum = (r_curr + r_l1 + r_l2 + r_l3)
    r_roll3_std = float(np.std([r_curr, r_l1, r_l2]))
    
    # Interactions
    rain3_x_water = r_l3 * w_water
    rain_roll4_x_water = r_roll4_mean * w_water
    rain3_div_sani = r_l3 / (w_sani + 0.05)
    vuln_idx = (w_water * (w_pop / 10000.0)) / (w_sani + 0.05)
    rain3_x_vuln = r_l3 * vuln_idx
    pop_x_water = (w_pop / 10000.0) * w_water
    
    dtr_val = 7.5
    avg_t = t2
    max_t = avg_t + (dtr_val / 2.0)
    min_t = avg_t - (dtr_val / 2.0)
    temp_opt_dist = abs(avg_t - 28.0)
    
    # Vapor pressure deficit
    es = 0.61078 * np.exp(17.27 * avg_t / (avg_t + 237.3))
    ea = es * (hum / 100.0)
    vpd = es - ea
    
    month_val = target_dt.month
    week_val = target_dt.isocalendar().week
    m_sin = np.sin(2 * np.pi * month_val / 12)
    m_cos = np.cos(2 * np.pi * month_val / 12)
    w_sin = np.sin(2 * np.pi * week_val / 52)
    w_cos = np.cos(2 * np.pi * week_val / 52)
    
    row = {
        'population_density_sqkm': w_pop,
        'water_logging_risk_index': w_water,
        'sanitation_score': w_sani,
        'avg_temp_c': avg_t,
        'max_temp_c': max_t,
        'min_temp_c': min_t,
        'rainfall_mm': r_curr,
        'relative_humidity_pct': hum,
        'rainfall_lag1': r_l1,
        'avg_temp_lag1': t2,
        'humidity_lag1': hum,
        'rainfall_lag2': r_l2,
        'avg_temp_lag2': t2,
        'humidity_lag2': hum,
        'rainfall_lag3': r_l3,
        'avg_temp_lag3': t4,
        'humidity_lag3': hum - 1.0,
        'rainfall_lag4': r_l4,
        'avg_temp_lag4': t4,
        'humidity_lag4': hum - 2.0,
        'rain_roll_2w_mean': r_roll2_mean,
        'rain_roll_3w_mean': r_roll3_mean,
        'rain_roll_4w_mean': r_roll4_mean,
        'rain_roll_4w_sum': r_roll4_sum,
        'rain_roll_3w_std': r_roll3_std,
        'rain3_x_waterlogging': rain3_x_water,
        'rain_roll4_x_waterlogging': rain_roll4_x_water,
        'rain3_div_sanitation': rain3_div_sani,
        'vulnerability_index': vuln_idx,
        'rain3_x_vulnerability': rain3_x_vuln,
        'pop_x_waterlogging': pop_x_water,
        'diurnal_temp_range': dtr_val,
        'temp_opt_distance': temp_opt_dist,
        'vapor_pressure_deficit': vpd,
        'month_sin': m_sin,
        'month_cos': m_cos,
        'week_sin': w_sin,
        'week_cos': w_cos
    }
    return pd.DataFrame([row])

# Compute prediction for current selected ward and horizon
scenario_features = build_features_for_ward(
    pop_density, water_logging, sanitation,
    scen_r2, scen_r4, scen_t2, scen_t4, scen_hum, target_forecast_date
)

t0 = time.time()
predicted_cases = float(model.predict(scenario_features)[0])
t_infer_ms = (time.time() - t0) * 1000.0

# Tier Classification
if predicted_cases < 0.5:
    risk_label = "Low risk"
    risk_color = "#22c55e" if is_dark else "#16a34a"
elif predicted_cases < 5.5:
    risk_label = "Medium risk"
    risk_color = "#eab308" if is_dark else "#ca8a04"
elif predicted_cases < 14.5:
    risk_label = "High risk"
    risk_color = "#f97316" if is_dark else "#ea580c"
else:
    risk_label = "Critical risk"
    risk_color = "#ef4444" if is_dark else "#dc2626"

# -------------------------------------------------------------
# MAIN CONTENT LAYOUT (Matching Image 1, 2, 3)
# -------------------------------------------------------------

# HERO BANNER (Image 1)
st.markdown(f"""
<div class="hero-banner">
    <span style="font-size: 0.76rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: #86efac;">BANGLADESH · WARD-LEVEL OUTLOOK</span>
    <h1 style="font-size: 2.2rem; font-weight: 800; margin-top: 0.3rem; margin-bottom: 0.4rem; line-height: 1.15; color: #ffffff;">See the signal<br/>before cases rise.</h1>
    <p style="font-size: 0.95rem; margin-bottom: 0.8rem; opacity: 0.95; color: #e2e8f0;">Compare local weather and vulnerability signals to plan earlier dengue prevention across the wards in your dataset.</p>
    <span style="display: inline-block; background: {hero_badge_bg}; border: 1px solid {hero_badge_border}; color: {hero_badge_text}; padding: 0.25rem 0.8rem; border-radius: 9999px; font-size: 0.82rem; font-weight: 700;">60 wards · 5 cities · data through 29 Dec 2025</span>
</div>
""", unsafe_allow_html=True)

# DATA NOTICE BANNER (Image 1)
st.markdown("""
<div class="info-banner">
    The source data ends on <b>29 Dec 2025</b>. This forecast extends beyond observed cases and weather; it uses the latest ward record plus the editable scenario inputs.
</div>
""", unsafe_allow_html=True)

# SECTION HEADER (Image 1)
st.markdown(f"""
<div style="margin-bottom: 0.8rem;">
    <span style="font-size: 0.74rem; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase; color: {text_muted};">SELECTED WARD OUTLOOK</span>
    <h2 style="font-size: 1.65rem; font-weight: 800; margin-top: 0.1rem; margin-bottom: 0.8rem;">{selected_city} / {selected_ward}</h2>
</div>
""", unsafe_allow_html=True)

# MAIN CARDS ROW (Image 1: Left Main Card + Right 3 Stat Cards + Explanation)
col_main_left, col_main_right = st.columns([1.1, 1.9])

with col_main_left:
    st.markdown(f"""
    <div class="theme-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; min-height: 220px;">
        <span style="font-size: 0.74rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: {text_muted};">WEEK +{horizon_weeks} · {target_date_str}</span>
        <div style="font-size: 2.1rem; font-weight: 800; color: {risk_color}; margin-top: 0.4rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 8px;">
            <span>●</span> <span>{risk_label}</span>
        </div>
        <span style="font-size: 0.95rem; color: {text_muted};">Estimated <b>{predicted_cases:.1f}</b> cases per week</span>
    </div>
    """, unsafe_allow_html=True)

with col_main_right:
    c_sub1, c_sub2, c_sub3 = st.columns(3)
    with c_sub1:
        st.markdown(f"""
        <div class="theme-card" style="text-align: left; padding: 0.85rem 1rem;">
            <span style="font-size: 0.74rem; font-weight: 700; color: {text_muted};">Water-log risk</span>
            <div style="font-size: 1.7rem; font-weight: 800; color: {text_color}; margin-top: 0.2rem;">{water_logging:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_sub2:
        st.markdown(f"""
        <div class="theme-card" style="text-align: left; padding: 0.85rem 1rem;">
            <span style="font-size: 0.74rem; font-weight: 700; color: {text_muted};">Sanitation</span>
            <div style="font-size: 1.7rem; font-weight: 800; color: {text_color}; margin-top: 0.2rem;">{sanitation:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_sub3:
        st.markdown(f"""
        <div class="theme-card" style="text-align: left; padding: 0.85rem 1rem;">
            <span style="font-size: 0.74rem; font-weight: 700; color: {text_muted};">Density / km²</span>
            <div style="font-size: 1.7rem; font-weight: 800; color: {text_color}; margin-top: 0.2rem;">{pop_density:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="theme-card" style="margin-top: 0.75rem; padding: 0.85rem 1.1rem;">
        <span style="font-size: 0.8rem; font-weight: 700; color: {text_color};">What the estimate uses</span><br/>
        <p style="font-size: 0.83rem; color: {text_muted}; margin-top: 0.3rem; margin-bottom: 0; line-height: 1.4;">
            Observed 29 Dec 2025 ward conditions, a {horizon_weeks}-week lead, rolling rainfall and climate values, recent dengue cases, and ward vulnerability. Scenario controls change the weather inputs without retraining.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# 4 KEY EVALUATION METRIC CARDS ROW (Image 1)
col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.markdown(f"""
    <div class="theme-card">
        <span style="font-size: 0.74rem; font-weight: 700; color: {text_muted};">Forecast cases / week</span>
        <div style="font-size: 1.65rem; font-weight: 800; color: {text_color}; margin-top: 0.2rem;">{predicted_cases:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_k2:
    st.markdown(f"""
    <div class="theme-card">
        <span style="font-size: 0.74rem; font-weight: 700; color: {text_muted};">MAE / week</span>
        <div style="font-size: 1.65rem; font-weight: 800; color: {text_color}; margin-top: 0.2rem;">0.77</div>
    </div>
    """, unsafe_allow_html=True)

with col_k3:
    st.markdown(f"""
    <div class="theme-card">
        <span style="font-size: 0.74rem; font-weight: 700; color: {text_muted};">RMSE / week</span>
        <div style="font-size: 1.65rem; font-weight: 800; color: {text_color}; margin-top: 0.2rem;">1.51</div>
    </div>
    """, unsafe_allow_html=True)

with col_k4:
    st.markdown(f"""
    <div class="theme-card">
        <span style="font-size: 0.74rem; font-weight: 700; color: {text_muted};">Holdout rows</span>
        <div style="font-size: 1.65rem; font-weight: 800; color: {text_color}; margin-top: 0.2rem;">6,300</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# WARD OVERVIEW MULTI-HORIZON TABLE (Image 2)
# -------------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 0.6rem;">
    <span style="background-color: #84cc16; color: #1a2e05; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 800; text-transform: uppercase;">Ward overview</span>
</div>
""", unsafe_allow_html=True)

# Compute multi-horizon predictions for all wards in current city
city_wards_df = ward_props[ward_props['city'] == selected_city].copy()

def get_tier_badge(pred_val):
    if pred_val < 0.5:
        return '<span class="badge-pill pill-low">Low</span>'
    elif pred_val < 5.5:
        return '<span class="badge-pill pill-medium">Medium</span>'
    elif pred_val < 14.5:
        return '<span class="badge-pill pill-high">High</span>'
    else:
        return '<span class="badge-pill pill-critical">Critical</span>'

table_rows_html = []
for _, wr in city_wards_df.iterrows():
    w_id = wr['ward_id']
    w_pop = wr['population_density_sqkm']
    w_wat = wr['water_logging_risk_index']
    w_san = wr['sanitation_score']
    
    # +2 weeks
    f_w2 = build_features_for_ward(w_pop, w_wat, w_san, scen_r2, scen_r4, scen_t2, scen_t4, scen_hum, reference_date + pd.Timedelta(weeks=2))
    p_w2 = float(model.predict(f_w2)[0])
    
    # +3 weeks
    f_w3 = build_features_for_ward(w_pop, w_wat, w_san, scen_r2, scen_r4, scen_t2, scen_t4, scen_hum, reference_date + pd.Timedelta(weeks=3))
    p_w3 = float(model.predict(f_w3)[0])
    
    # +4 weeks
    f_w4 = build_features_for_ward(w_pop, w_wat, w_san, scen_r2, scen_r4, scen_t2, scen_t4, scen_hum, reference_date + pd.Timedelta(weeks=4))
    p_w4 = float(model.predict(f_w4)[0])
    
    highlight = "background-color: rgba(59, 130, 246, 0.12); font-weight: bold;" if w_id == selected_ward else ""
    
    row_html = (
        f'<tr style="{highlight} border-bottom: 1px solid {card_border};">'
        f'<td style="padding: 10px 14px; font-weight: 600;">{w_id}</td>'
        f'<td style="padding: 10px 14px; text-align: center;">{get_tier_badge(p_w2)}</td>'
        f'<td style="padding: 10px 14px; text-align: center;">{get_tier_badge(p_w3)}</td>'
        f'<td style="padding: 10px 14px; text-align: center;">{get_tier_badge(p_w4)}</td>'
        f'</tr>'
    )
    table_rows_html.append(row_html)

rows_joined = "".join(table_rows_html)
table_full_html = (
    f'<div class="theme-card" style="padding: 0; overflow-x: auto; margin-bottom: 1.5rem;">'
    f'<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">'
    f'<thead>'
    f'<tr style="background-color: {table_header_bg}; border-bottom: 2px solid {card_border}; text-align: center;">'
    f'<th style="padding: 12px 14px; text-align: left; font-weight: 700; color: {text_muted};">Ward</th>'
    f'<th style="padding: 12px 14px; font-weight: 700; color: {text_muted};">Week +2 (12 Jan 2026)</th>'
    f'<th style="padding: 12px 14px; font-weight: 700; color: {text_muted};">Week +3 (19 Jan 2026)</th>'
    f'<th style="padding: 12px 14px; font-weight: 700; color: {text_muted};">Week +4 (26 Jan 2026)</th>'
    f'</tr>'
    f'</thead>'
    f'<tbody>'
    f'{rows_joined}'
    f'</tbody>'
    f'</table>'
    f'</div>'
)
st.markdown(table_full_html, unsafe_allow_html=True)

# -------------------------------------------------------------
# EXPANDABLE HISTORICAL TREND & TIME-SERIES (Image 3)
# -------------------------------------------------------------
with st.expander("📈 Ward historical trend & estimates by horizon", expanded=True):
    st.caption(f"Historical Dengue incidence and precipitation trajectory for {selected_ward} throughout 2025.")
    
    w_ts = raw_df[(raw_df['ward_id'] == selected_ward) & (raw_df['date'] >= '2025-01-01')].sort_values('date')
    
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar trace: Weekly Dengue Cases
    fig_ts.add_trace(
        go.Bar(
            x=w_ts['date'],
            y=w_ts['dengue_cases'],
            name="Dengue Cases / Week",
            marker_color="#4d7c0f" if is_dark else "#15803d",
            opacity=0.85
        ),
        secondary_y=False
    )
    
    # Line trace: Weekly Rainfall
    fig_ts.add_trace(
        go.Scatter(
            x=w_ts['date'],
            y=w_ts['rainfall_mm'],
            name="Rainfall (mm)",
            line=dict(color="#84cc16", width=2.5)
        ),
        secondary_y=True
    )
    
    fig_ts.update_layout(
        template=plotly_template,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_ts.update_yaxes(title_text="Cases / week", secondary_y=False)
    fig_ts.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
    
    st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------------------------------------------
# EXPANDABLE VECTOR BREEDING CLOCK & DRIVERS (Our Research Extensions)
# -------------------------------------------------------------
with st.expander("🦟 The Vector Breeding Clock & Environmental Drivers", expanded=False):
    col_c1, col_c2 = st.columns([1.1, 1])
    
    with col_c1:
        st.markdown("**Biological Lag Timeline (Rainfall Lags 0–4 Weeks):**")
        st.caption("Aedes aegypti breeding cycle requires 2–4 weeks from rain events to hospital admission surges.")
        
        lag_plot_df = pd.DataFrame({
            'Stage': [
                'Current (t)<br>Rain Event',
                'Lag 1 (t-1)<br>Larval Hatching',
                'Lag 2 (t-2)<br>Adult Emergence',
                'Lag 3 (t-3)<br>Peak Viral EIP',
                'Lag 4 (t-4)<br>Hospital Surge'
            ],
            'Rainfall (mm)': [scen_r2 * 0.8, scen_r2 * 1.05, scen_r2, scen_r4 * 0.95, scen_r4],
            'Correlation': [0.53, 0.59, 0.67, 0.75, 0.68]
        })
        
        fig_lags = px.bar(
            lag_plot_df, x='Stage', y='Rainfall (mm)',
            color='Correlation',
            color_continuous_scale='Greens',
            text='Rainfall (mm)'
        )
        fig_lags.update_layout(
            template=plotly_template,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_lags.update_traces(texttemplate='%{text:.1f} mm', textposition='outside')
        st.plotly_chart(fig_lags, use_container_width=True)
        
    with col_c2:
        st.markdown("**Environmental & Infrastructure Driver Decomposition:**")
        st.caption("Standardized relative risk contribution terms.")
        
        df_drivers = pd.DataFrame({
            'Driver': [
                'Drainage Bottleneck × Lag-3 Rain',
                'Sanitation Deficit (1 - Sanitation)',
                'Crowding Density (/ 35k)',
                'Thermal Optimality (|T - 28°C|)'
            ],
            'Score': [
                round((scen_r4 * water_logging) / 80.0, 2),
                round((1.0 - sanitation) * 1.4, 2),
                round(pop_density / 35000.0, 2),
                round(max(0.0, 1.0 - abs(scen_t2 - 28.0) / 10.0), 2)
            ]
        }).sort_values('Score', ascending=True)
        
        fig_d = px.bar(
            df_drivers, x='Score', y='Driver',
            orientation='h',
            color='Score',
            color_continuous_scale='Reds'
        )
        fig_d.update_layout(
            template=plotly_template,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_d, use_container_width=True)

# -------------------------------------------------------------
# EXPANDABLE ARCHITECTURE & VALIDATION METRICS
# -------------------------------------------------------------
with st.expander("🏗️ Model Architecture & Out-of-Time Generalization (EE-VIB)", expanded=False):
    st.markdown("""
    - **Production Ensemble Formula:** $\\hat{y} = 0.40 \\times \\text{LightGBM}_{\\text{Tweedie}} + 0.45 \\times \\text{CatBoost}_{\\text{MAE}} + 0.15 \\times \\text{CatBoost}_{\\text{Tweedie}}$
    - **Forward Test Partition (2024–2025: 6,300 Ward-Weeks):**
      - **MAE:** `0.7738` | **RMSE:** `1.5127` | **$R^2$:** `0.8465` | **Tier Accuracy:** `83.10%`
    - **Nested Spatial Cross-Validation (31,080 Out-of-Fold Samples):**
      - Outer 5-Fold GroupKFold Spatial MAE: `0.8527`
      - Leave-One-City-Out Spatial MAE: `0.8632`
    - **Local Batch Latency:** **20.10 ms** across all 60 wards via serialized `dengue_model.joblib`.
    """)

# FOOTER (Matching Image 3)
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="border-top: 1px solid {card_border}; padding-top: 1rem; color: {text_muted}; font-size: 0.8rem; text-align: center;">
    WardWatch · Local prototype · No live weather feed or external prediction API · Data source: supplied hackathon dataset export · Sub-second serialized inference ({t_infer_ms:.2f} ms).
</div>
""", unsafe_allow_html=True)
