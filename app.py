import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

st.set_page_config(
    page_title="BioGuard MRO",
    page_icon="🛩️",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('bioguard_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv('df_clean.csv')

model, feature_cols = load_model()
df = load_data()

st.markdown("""
    <h1 style='text-align:center; color:#1E90FF;'>
    🛩️ BioGuard MRO Dashboard
    </h1>
    <p style='text-align:center; color:gray;'>
    AI-Powered Engine Health Monitoring
    </p><hr>
""", unsafe_allow_html=True)

# Engine select + Speed
col_a, col_b = st.columns([2, 1])
with col_a:
    engine_ids = sorted(df['engine_id'].unique())
    selected = st.selectbox("Select Engine:",
                            engine_ids,
                            format_func=lambda x: f"Engine {int(x)}")
with col_b:
    delay = st.slider("Simulation Speed :",
                      1, 100, 10)

engine_data = df[df['engine_id']==selected].reset_index(drop=True)

start = st.button("▶️ Start Simulation", type="primary")

st.divider()

# Placeholders
status_placeholder = st.empty()
col1, col2, col3   = st.columns(3)
cycle_ph  = col1.empty()
status_ph = col2.empty()
risk_ph   = col3.empty()

st.divider()
st.markdown("### 📈 Live Sensor Trends")
chart_ph = st.empty()

label_map = {0:'🟢 SAFE',   1:'🟡 WARNING',  2:'🔴 CRITICAL'}
color_hex = {0:'#2ecc71',   1:'#f39c12',     2:'#e74c3c'}
sensors   = [
    ('s4', 'LPT Temperature', 'navy'),
    ('s7', 'HPC Pressure',    'darkgreen'),
    ('s9', 'Core Speed',      'darkred'),
]

patches = [
    mpatches.Patch(color='#2ecc71', label='SAFE'),
    mpatches.Patch(color='#f39c12', label='WARNING'),
    mpatches.Patch(color='#e74c3c', label='CRITICAL'),
]

if start:
    cycles_so_far = []
    preds_so_far  = []
    s4_so_far     = []
    s7_so_far     = []
    s9_so_far     = []

    for i, row in engine_data.iterrows():

        # Predict
        X_row = pd.DataFrame([row[feature_cols]])
        pred  = model.predict(X_row)[0]
        prob  = model.predict_proba(X_row)[0]
        risk  = round(prob[2] * 100, 1)

        cycles_so_far.append(int(row['time_cycle']))
        preds_so_far.append(pred)
        s4_so_far.append(row['s4'])
        s7_so_far.append(row['s7'])
        s9_so_far.append(row['s9'])

        # Status update — har cycle
        if pred == 2:
            status_placeholder.error(
                f"🔴 Cycle {int(row['time_cycle'])} — CRITICAL — Ground Immediately!")
        elif pred == 1:
            status_placeholder.warning(
                f"🟡 Cycle {int(row['time_cycle'])} — WARNING — Schedule Maintenance!")
        else:
            status_placeholder.success(
                f"🟢 Cycle {int(row['time_cycle'])} — SAFE — All Good!")

        # Metrics — har cycle
        cycle_ph.metric("Current Cycle",  int(row['time_cycle']))
        status_ph.metric("Status",        label_map[pred])
        risk_ph.metric("Risk Score",      f"{risk}%")

        # Graph — har 5 cycles pe update
        if i % 5 == 0 or i == len(engine_data)-1:
            fig, axes = plt.subplots(3, 1, figsize=(12, 9))

            data_sets    = [s4_so_far, s7_so_far, s9_so_far]
            point_colors = [color_hex[p] for p in preds_so_far]

            for j, (sensor_vals, (sensor, name, color)) in \
                    enumerate(zip(data_sets, sensors)):
                ax = axes[j]
                ax.plot(cycles_so_far, sensor_vals,
                        color=color, linewidth=1.5, alpha=0.7)
                ax.scatter(cycles_so_far, sensor_vals,
                           c=point_colors, s=30, zorder=5)
                ax.set_title(f'{name} ({sensor})',
                             fontweight='bold')
                ax.set_ylabel(name)
                ax.set_xlabel('Cycle')
                ax.set_xlim(0, engine_data['time_cycle'].max()+5)

            # Legend — top center
            fig.legend(handles=patches,
                       loc='upper center',
                       ncol=3,
                       bbox_to_anchor=(0.5, 1.02),
                       fontsize=11,
                       framealpha=0.8)

            plt.tight_layout()
            chart_ph.pyplot(fig)
            plt.close()

        time.sleep(delay / 1000)

    # End
    if preds_so_far[-1] == 2:
        status_placeholder.error("💥 ENGINE FAILED")
    else:
        status_placeholder.success("✅ Simulation Complete!")
