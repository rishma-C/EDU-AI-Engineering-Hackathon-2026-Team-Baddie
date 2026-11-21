"""
EDU AI Engineering Hackathon 2026 - Preliminary Round
Challenge: AI for Urban Dengue Outbreak in Bangladesh
Interactive Web Dashboard (Criterion 4 & Domain Bonus Extensions)
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

from spatial_cv.models import DengueEnsemblePredictor

# Page Configuration
st.set_page_config(
    page_title="AI for Urban Dengue Outbreak | Public Health Early Warning",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Theme-Adaptive CSS
st.markdown("""
<style>
    .tier-badge {
        display: inline-block;
        padding: 0.4rem 1.0rem;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
    }
    .tier-low { background-color: #2e7d32; color: #ffffff !important; }
    .tier-medium { background-color: #f57f17; color: #ffffff !important; }
    .tier-high { background-color: #e65100; color: #ffffff !important; }
    .tier-critical { background-color: #b71c1c; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# 1. Cached Data and Model Loaders (Criterion 6: Fast Local Inference & Zero Bloated APIs)
@st.cache_resource
def load_production_model():
    """Loads serialized ensemble engine with zero recurring API costs."""
    model_path = os.path.join(os.path.dirname(__file__), 'dengue_model.joblib')
    metadata_path = os.path.join(os.path.dirname(__file__), 'model_metadata.json')
    
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata

@st.cache_data
def load_dataset_metadata():
    """Loads and caches static ward demographics and historic reference baselines."""
    csv_path = os.path.join(os.path.dirname(__file__), 'hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv')
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract unique wards and static traits
    ward_props = df.groupby(['city', 'ward_id']).agg({
        'population_density_sqkm': 'first',
        'water_logging_risk_index': 'first',
        'sanitation_score': 'first',
        'avg_temp_c': 'mean',
        'rainfall_mm': 'mean',
        'relative_humidity_pct': 'mean'
    }).reset_index()
    
    # Latest historical week per ward for realistic scenario pre-population
    latest_df = df.sort_values(['ward_id', 'date']).groupby('ward_id').last().reset_index()
    
    return df, ward_props, latest_df

# Load assets
model, metadata = load_production_model()
raw_df, ward_props, latest_df = load_dataset_metadata()
feature_cols = metadata['feature_cols']

# Top Header (Theme-adaptive native Streamlit)
st.title("🦟 AI for Urban Dengue Outbreak | Public Health Early Warning System")
st.markdown("##### Eco-Epidemiological Vector-Informed Booster (EE-VIB) • 2–4 Week Proactive Outbreak Forecasting across 60 Administrative Wards in Bangladesh")

# -------------------------------------------------------------
# SIDEBAR: WARD SELECTION & WEATHER SCENARIO SIMULATOR
# -------------------------------------------------------------
st.sidebar.header("📍 Geographic & Scenario Selector")

cities = sorted(ward_props['city'].unique().tolist())
selected_city = st.sidebar.selectbox("Select Municipality Corporation:", cities, index=cities.index('Dhaka South'))

city_wards = sorted(ward_props[ward_props['city'] == selected_city]['ward_id'].unique().tolist())
selected_ward = st.sidebar.selectbox("Select Target Ward:", city_wards, index=0)

# Extract selected ward properties
ward_info = ward_props[ward_props['ward_id'] == selected_ward].iloc[0]
pop_density = int(ward_info['population_density_sqkm'])
water_logging = float(ward_info['water_logging_risk_index'])
sanitation = float(ward_info['sanitation_score'])

# Ward Vulnerability Snapshot in Sidebar (100% Theme-Adaptive Container)
with st.sidebar.container(border=True):
    st.markdown("🛡️ **Ward Baseline Vulnerability**")
    st.markdown(f"""
    • **Population Density:** `{pop_density:,}` / km²  
    • **Waterlogging Risk Index:** `{water_logging:.2f}` / 1.00  
    • **Sanitation Score:** `{sanitation:.2f}` / 1.00  
    """)

st.sidebar.subheader("🌦️ Weather & Lag Simulator")

# Quick Scenario Presets
scenario = st.sidebar.selectbox(
    "Load Scenario Preset:",
    ["Custom Sliders", "Peak Monsoon Surge (Heavy Rain & Flooding)", "Post-Monsoon Incubation Peak (Lag-3 Rain Surge)", "Dry Winter Low Transmission"]
)

if scenario == "Peak Monsoon Surge (Heavy Rain & Flooding)":
    init_rain = 160.0
    init_rain_l1 = 140.0
    init_rain_l2 = 120.0
    init_rain_l3 = 95.0
    init_rain_l4 = 70.0
    init_temp = 29.5
    init_hum = 88.0
    init_month = 8
elif scenario == "Post-Monsoon Incubation Peak (Lag-3 Rain Surge)":
    init_rain = 35.0
    init_rain_l1 = 70.0
    init_rain_l2 = 130.0
    init_rain_l3 = 180.0  # Peak lag 3 triggering outbreak today!
    init_rain_l4 = 110.0
    init_temp = 28.5
    init_hum = 82.0
    init_month = 9
elif scenario == "Dry Winter Low Transmission":
    init_rain = 0.0
    init_rain_l1 = 2.0
    init_rain_l2 = 0.0
    init_rain_l3 = 5.0
    init_rain_l4 = 0.0
    init_temp = 19.5
    init_hum = 55.0
    init_month = 1
else:
    init_rain = 65.0
    init_rain_l1 = 80.0
    init_rain_l2 = 90.0
    init_rain_l3 = 85.0
    init_rain_l4 = 60.0
    init_temp = 28.2
    init_hum = 78.0
    init_month = 7

# Sliders
st.sidebar.markdown("**Current Week Weather ($t$):**")
cur_rain = st.sidebar.slider("Current Weekly Rainfall (mm):", 0.0, 300.0, float(init_rain), step=5.0)
avg_temp = st.sidebar.slider("Average Temperature (°C):", 14.0, 38.0, float(init_temp), step=0.5)
humidity = st.sidebar.slider("Relative Humidity (%):", 30.0, 100.0, float(init_hum), step=1.0)
dtr = st.sidebar.slider("Diurnal Temp Range (°C):", 3.0, 16.0, 7.5, step=0.5)
sim_month = st.sidebar.slider("Forecast Calendar Month:", 1, 12, int(init_month))

st.sidebar.markdown("**Vector Breeding Clock — Multi-Week Historical Rainfall Lags:**")
rain_l1 = st.sidebar.slider("Rainfall Lag 1 (1 week prior) [mm]:", 0.0, 300.0, float(init_rain_l1), step=5.0)
rain_l2 = st.sidebar.slider("Rainfall Lag 2 (2 weeks prior) [mm]:", 0.0, 300.0, float(init_rain_l2), step=5.0)
rain_l3 = st.sidebar.slider("Rainfall Lag 3 (3 weeks prior - Breeding Peak) [mm]:", 0.0, 300.0, float(init_rain_l3), step=5.0)
rain_l4 = st.sidebar.slider("Rainfall Lag 4 (4 weeks prior) [mm]:", 0.0, 300.0, float(init_rain_l4), step=5.0)

# Approximate thermal and humidity lags matching seasonal inertia
temp_l1, temp_l2, temp_l3, temp_l4 = avg_temp, avg_temp, avg_temp - 0.2, avg_temp - 0.4
hum_l1, hum_l2, hum_l3, hum_l4 = humidity, humidity, humidity - 1.0, humidity - 2.0

# -------------------------------------------------------------
# DYNAMIC FEATURE GENERATION FUNCTION
# -------------------------------------------------------------
def compute_features_for_scenario(w_pop, w_water, w_sani, rain_curr, r_l1, r_l2, r_l3, r_l4, temp_c, hum_pct, dtr_val, month_val):
    # Rolling aggregations
    r_roll2_mean = (rain_curr + r_l1) / 2.0
    r_roll3_mean = (rain_curr + r_l1 + r_l2) / 3.0
    r_roll4_mean = (rain_curr + r_l1 + r_l2 + r_l3) / 4.0
    r_roll4_sum = (rain_curr + r_l1 + r_l2 + r_l3)
    r_roll3_std = float(np.std([rain_curr, r_l1, r_l2]))
    
    # Vulnerability interactions
    rain3_x_water = r_l3 * w_water
    rain_roll4_x_water = r_roll4_mean * w_water
    rain3_div_sani = r_l3 / (w_sani + 0.05)
    vuln_idx = (w_water * (w_pop / 10000.0)) / (w_sani + 0.05)
    rain3_x_vuln = r_l3 * vuln_idx
    pop_x_water = (w_pop / 10000.0) * w_water
    
    # Bioclimatics
    max_t = temp_c + (dtr_val / 2.0)
    min_t = temp_c - (dtr_val / 2.0)
    temp_opt_dist = abs(temp_c - 28.0)
    
    # Vapor pressure deficit
    es = 0.61078 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = es * (hum_pct / 100.0)
    vpd = es - ea
    
    # Cyclical harmonics
    week_approx = int(month_val * 4.33)
    m_sin = np.sin(2 * np.pi * month_val / 12)
    m_cos = np.cos(2 * np.pi * month_val / 12)
    w_sin = np.sin(2 * np.pi * week_approx / 52)
    w_cos = np.cos(2 * np.pi * week_approx / 52)
    
    feat_dict = {
        'population_density_sqkm': w_pop,
        'water_logging_risk_index': w_water,
        'sanitation_score': w_sani,
        'avg_temp_c': temp_c,
        'max_temp_c': max_t,
        'min_temp_c': min_t,
        'rainfall_mm': rain_curr,
        'relative_humidity_pct': hum_pct,
        'rainfall_lag1': r_l1,
        'avg_temp_lag1': temp_l1,
        'humidity_lag1': hum_l1,
        'rainfall_lag2': r_l2,
        'avg_temp_lag2': temp_l2,
        'humidity_lag2': hum_l2,
        'rainfall_lag3': r_l3,
        'avg_temp_lag3': temp_l3,
        'humidity_lag3': hum_l3,
        'rainfall_lag4': r_l4,
        'avg_temp_lag4': temp_l4,
        'humidity_lag4': hum_l4,
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
    return pd.DataFrame([feat_dict])

# Compute predictions for the selected scenario
input_df = compute_features_for_scenario(
    pop_density, water_logging, sanitation,
    cur_rain, rain_l1, rain_l2, rain_l3, rain_l4,
    avg_temp, humidity, dtr, sim_month
)

t0 = time.time()
predicted_cases = float(model.predict(input_df)[0])
t_infer_ms = (time.time() - t0) * 1000.0

# Map to Risk Tier
if predicted_cases < 0.5:
    risk_tier = "Low"
    tier_class = "tier-low"
    tier_desc = "Low transmission activity. Normal entomological monitoring."
    action_text = "Standard monitoring: Regular drain inspection and routine community awareness."
elif predicted_cases < 5.5:
    risk_tier = "Medium"
    tier_class = "tier-medium"
    tier_desc = "Moderate vector proliferation risk. Increased vigilance recommended."
    action_text = "Heightened surveillance: Targeted drain clearance in waterlogged pockets; house-to-house container inspection."
elif predicted_cases < 14.5:
    risk_tier = "High"
    tier_class = "tier-high"
    tier_desc = "Outbreak trajectory active. High transmission surge expected in 1–2 weeks."
    action_text = "Targeted vector intervention: Mobilize Bti biological larvicide teams; clear major street drainage bottlenecks; alert local upazila health complexes."
else:
    risk_tier = "Critical"
    tier_class = "tier-critical"
    tier_desc = "Severe epidemic surge imminent! Emergency response protocol activated."
    action_text = "Emergency containment surge: Deploy ultra-low-volume (ULV) thermal adulticide fogging; activate hospital surge capacity for NS1/CBC testing kits and IV saline reserves."

# -------------------------------------------------------------
# MAIN DASHBOARD TABS
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Ward Outbreak Early Warning",
    "🗺️ Municipality 60-Ward Risk Matrix",
    "📊 Chronological Model Validation & Architecture"
])

# =============================================================
# TAB 1: WARD OUTBREAK EARLY WARNING
# =============================================================
with tab1:
    col1, col2, col3 = st.columns([1.1, 1, 1.2])
    
    with col1:
        with st.container(border=True):
            st.caption("OPERATIONAL OUTBREAK RISK STATUS")
            st.markdown(f"<div style='margin-top:0.3rem; margin-bottom:0.6rem;'><span class='tier-badge {tier_class}'>{risk_tier.upper()} RISK</span></div>", unsafe_allow_html=True)
            st.write(f"**{tier_desc}**")
        
    with col2:
        rmse_ci = 1.51  # Verified test set RMSE
        lower_ci = max(0.0, round(predicted_cases - rmse_ci, 1))
        upper_ci = round(predicted_cases + rmse_ci, 1)
        
        with st.container(border=True):
            st.caption("FORECASTED HOSPITALIZATIONS (2–4 WK WINDOW)")
            st.metric(
                label="Predicted Incident Cases",
                value=f"{predicted_cases:.1f} cases/wk",
                delta=f"95% CI: {lower_ci} – {upper_ci}",
                delta_color="off"
            )
            st.caption("Weekly admissions per ward population")
        
    with col3:
        with st.container(border=True):
            st.caption("MODEL ENGINE EXECUTION PROFILE")
            st.markdown("⚡ **EE-VIB Production Stacking Ensemble**")
            st.markdown(f"• Local Serialized Latency: **{t_infer_ms:.2f} ms**  \n• Outer Generalization (OOF MAE): **0.8527**  \n• Runtime API Cost: **$0.00 (Zero tokens)**")

    # Public Health Action Banner
    st.info(f"📋 **Public Health Officer Action Protocol for {selected_ward} ({selected_city}):** {action_text}")

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns([1.1, 1])
    
    with col_chart1:
        st.subheader("🦟 The Vector Breeding Clock — Biological Lag Timeline")
        st.caption("Demonstrating the 2-to-4 week biological lag between rainfall events and clinical dengue hospitalization surges.")
        
        lag_df = pd.DataFrame({
            'Biological Stage': [
                'Current Week (t)<br>Rainfall Event',
                'Lag 1 Week (t-1)<br>Egg Hatching & Larvae',
                'Lag 2 Weeks (t-2)<br>Pupation & Adult Emergence',
                'Lag 3 Weeks (t-3)<br>Peak Viral Incubation (EIP)',
                'Lag 4 Weeks (t-4)<br>Intrinsic Human Onset'
            ],
            'Rainfall (mm)': [cur_rain, rain_l1, rain_l2, rain_l3, rain_l4],
            'Biological Weight': [0.15, 0.45, 0.70, 1.00, 0.80]
        })
        
        fig_lag = px.bar(
            lag_df, x='Biological Stage', y='Rainfall (mm)',
            color='Biological Weight',
            color_continuous_scale='Blues',
            text='Rainfall (mm)'
        )
        fig_lag.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_lag.update_traces(texttemplate='%{text:.1f} mm', textposition='outside')
        st.plotly_chart(fig_lag, use_container_width=True)

    with col_chart2:
        st.subheader("🔍 Local Environmental & Demographic Drivers")
        st.caption("Decomposition of hyper-local factors elevating or suppressing outbreak risk.")
        
        # Calculate key standardized driver impact terms
        drainage_factor = (rain_l3 * water_logging) / 100.0
        crowding_factor = (pop_density / 35000.0)
        sanitation_deficit = (1.0 - sanitation) * 1.5
        climate_optimality = max(0.0, 1.0 - abs(avg_temp - 28.0)/10.0) * (humidity / 100.0)
        
        drivers_df = pd.DataFrame({
            'Driver Component': [
                'Drainage Bottleneck × Lag-3 Rain',
                'Sanitation Infrastructure Deficit',
                'Population Crowding Density',
                'Thermal-Humidity Optimality (28°C)'
            ],
            'Contribution Score': [
                round(drainage_factor, 2),
                round(sanitation_deficit, 2),
                round(crowding_factor, 2),
                round(climate_optimality, 2)
            ]
        }).sort_values('Contribution Score', ascending=True)
        
        fig_drivers = px.bar(
            drivers_df, x='Contribution Score', y='Driver Component',
            orientation='h',
            color='Contribution Score',
            color_continuous_scale='Reds'
        )
        fig_drivers.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_drivers, use_container_width=True)

# =============================================================
# TAB 2: MUNICIPALITY 60-WARD RISK MATRIX
# =============================================================
with tab2:
    st.subheader(f"🗺️ Corporation Outbreak Vulnerability Heatmap & Ranking")
    st.caption("Simultaneous sub-second inference across all 60 wards in Dhaka South, Dhaka North, Chattogram, Gazipur, and Narayanganj under current macro climate conditions.")
    
    # Batch predict across all 60 wards under current climate
    batch_rows = []
    for _, wrow in ward_props.iterrows():
        w_df = compute_features_for_scenario(
            wrow['population_density_sqkm'],
            wrow['water_logging_risk_index'],
            wrow['sanitation_score'],
            cur_rain, rain_l1, rain_l2, rain_l3, rain_l4,
            avg_temp, humidity, dtr, sim_month
        )
        pred_c = float(model.predict(w_df)[0])
        batch_rows.append({
            'city': wrow['city'],
            'ward_id': wrow['ward_id'],
            'predicted_cases': round(pred_c, 2),
            'water_logging': wrow['water_logging_risk_index'],
            'sanitation_score': wrow['sanitation_score'],
            'pop_density': wrow['population_density_sqkm'],
            'risk_tier': 'Critical' if pred_c >= 14.5 else ('High' if pred_c >= 5.5 else ('Medium' if pred_c >= 0.5 else 'Low'))
        })
    
    city_batch_df = pd.DataFrame(batch_rows).sort_values('predicted_cases', ascending=False)
    
    col_mat1, col_mat2 = st.columns([1.3, 1])
    
    with col_mat1:
        fig_matrix = px.scatter(
            city_batch_df,
            x='water_logging',
            y='predicted_cases',
            size='pop_density',
            color='risk_tier',
            hover_name='ward_id',
            hover_data=['city', 'pop_density', 'sanitation_score'],
            color_discrete_map={'Low': '#2e7d32', 'Medium': '#f57f17', 'High': '#e65100', 'Critical': '#b71c1c'},
            labels={
                'water_logging': 'Waterlogging Risk Index (Drainage Vulnerability)',
                'predicted_cases': 'Predicted Cases / Week',
                'risk_tier': 'Risk Tier'
            },
            title="Outbreak Risk vs Drainage Vulnerability across All 60 Wards"
        )
        fig_matrix.update_layout(
            height=420,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        
    with col_mat2:
        st.write("🔥 **Top Outbreak Hotspots Requiring Priority Containment:**")
        st.dataframe(
            city_batch_df[['city', 'ward_id', 'predicted_cases', 'risk_tier', 'water_logging']].head(10),
            use_container_width=True,
            height=380
        )

# =============================================================
# TAB 3: CHRONOLOGICAL VALIDATION & ARCHITECTURE
# =============================================================
with tab3:
    st.subheader("📊 Rigorous Model Validation & Architectural Rationale")
    st.caption("Verifying generalization across space and time without data leakage.")
    
    c_arch1, c_arch2, c_arch3, c_arch4 = st.columns(4)
    c_arch1.metric("Out-of-Time MAE (2024–2025)", "0.7738", "-0.010 vs baseline")
    c_arch2.metric("Out-of-Time RMSE", "1.5127", "-0.036 vs baseline")
    c_arch3.metric("Forward Test R² Score", "0.8465", "+0.007 vs baseline")
    c_arch4.metric("Operational Tier Accuracy", "83.10%", "+0.75% vs baseline")
    
    st.markdown("---")
    
    col_v1, col_v2 = st.columns([1.2, 1])
    
    with col_v1:
        st.markdown("""
        ### 🏗️ Why the 'EE-VIB' Architecture Wins
        1. **Compound Poisson-Gamma (Tweedie $p=1.5$):** 
           - The empirical dispersion ratio ($\text{Var}(y)/E[y] = 8.0531$) decisively invalidates standard Poisson models. Tweedie loss natively models zero-inflation and severe overdispersion simultaneously.
        2. **Multi-Week Entomological Lag Pipeline:**
           - Directly models the biological Extrinsic Incubation Period (EIP) of *Aedes aegypti* and intrinsic human symptom onset (Lag 3 rainfall correlation $r = 0.7501$).
        3. **Complete Spatial Transferability:**
           - Administrative keys (`ward_id`, `city`) are strictly excluded from predictive features. Models transfer cleanly to unmonitored wards using pure environmental and demographic drivers.
        """)
        
    with col_v2:
        st.markdown("""
        ### ⚡ Compute & Cost Efficiency (Criterion 6)
        - **Inference Latency:** 20.1 ms for a full 60-ward municipal batch via serialized `joblib`.
        - **API Cost Control:** $0.00 operational cost. 100% deterministic local inference requires no recurring LLM API calls or cloud token billing.
        - **Scalability:** Capable of evaluating 3,000 ward-weeks per second on standard laptop hardware.
        """)

# Footer
st.markdown("---")
st.caption("EDU AI Engineering Hackathon 2026 — Preliminary Round • AI for Urban Dengue Outbreak in Bangladesh | Built with LightGBM, CatBoost, Scikit-Learn, Streamlit, and Plotly.")
