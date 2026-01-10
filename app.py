import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.express as px
from streamlit_option_menu import option_menu

# --- 1. Page Configuration ---
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

working_dir = os.path.dirname(os.path.abspath(__file__))

# --- 2. Load Assets ---
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

@st.cache_resource
def load_assets():
    model = pickle.load(open(os.path.join(working_dir, 'parkinsons_model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(working_dir, 'scaler.pkl'), 'rb'))
    return model, scaler

try:
    parkinsons_model, scaler = load_assets()
except Exception as e:
    st.error(f"Missing Files: Ensure parkinsons_model.pkl and scaler.pkl are in the folder.")
    st.stop()

# --- 3. Initialize Session State ---
for feature in feature_names:
    if feature not in st.session_state:
        st.session_state[feature] = 0.0

if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# --- 4. Sidebar ---
with st.sidebar:
    selected = option_menu('Disease Prediction System',
                           ['Single Person Prediction', 'Entire Batch Prediction'],
                           menu_icon='hospital-fill', icons=['person-fill', 'file-earmark-spreadsheet-fill'],
                           default_index=0)
    
    st.divider()
    if st.button("🗑️ Reset All Data", use_container_width=True):
        for feature in feature_names:
            st.session_state[feature] = 0.0
        if 'batch_results' in st.session_state:
            del st.session_state.batch_results
        st.session_state.file_uploader_key += 1
        st.rerun()

# --- 5. Single Prediction Page ---
if selected == "Single Person Prediction":
    st.title("Parkinson's Disease Prediction")

    # FIXED QUICK FILL WITH DROP-DOWN
    with st.expander("📥 Quick Fill: Paste CSV Row", expanded=False):
        st.info("Paste a comma-separated row. This will instantly fill the 22 boxes below.")
        paste_input = st.text_area("Paste row here:", height=100)
        
        if st.button("🚀 Apply Data to Metrics", type="secondary"):
            if paste_input:
                try:
                    # Clean the string (handle quotes from Excel/CSV)
                    clean_input = paste_input.replace('"', '').replace("'", "").strip()
                    values = [float(v.strip()) for v in clean_input.split(',')]
                    
                    if len(values) == len(feature_names):
                        for i, feature in enumerate(feature_names):
                            st.session_state[feature] = values[i]
                        st.success("Values applied! Scroll down to see metrics.")
                        st.rerun() # This is critical to force the number_inputs to update
                    else:
                        st.error(f"Value Mismatch: Expected 22 values, found {len(values)}")
                except ValueError:
                    st.error("Format Error: Ensure the input contains only numbers and commas.")

    st.divider()
    
    # Input Grid
    st.markdown("### ✍️ Individual Voice Metrics")
    cols = st.columns(5)
    current_inputs = {} 

    for i, feature in enumerate(feature_names):
        with cols[i % 5]:
            # Linking value directly to session_state ensures instant update on rerun
            val = st.number_input(feature, step=0.00001, format="%.5f", 
                                  value=st.session_state[feature], 
                                  key=f"input_{feature}")
            current_inputs[feature] = val

    if st.button("Analyze Test Results", type="primary", use_container_width=True):
        input_df = pd.DataFrame([current_inputs])[feature_names]
        std_data = scaler.transform(input_df)
        prediction = parkinsons_model.predict(std_data)
        
        if hasattr(parkinsons_model, "predict_proba"):
            prob = parkinsons_model.predict_proba(std_data)[0][1]
            st.info(f"Model Confidence Score: {prob:.2%}")

        if prediction[0] == 1:
            st.error("Detection Result: High Probability of Parkinson's Disease.")
        else:
            st.success("Detection Result: Low Probability of Parkinson's Disease.")

# --- 6. Batch Prediction Page ---
elif selected == "Entire Batch Prediction":
    st.title("📊 Batch Diagnostic & Analytics Tool")
    
    uploaded_file = st.file_uploader("Upload CSV Data File", type=['csv'], 
                                     key=f"uploader_{st.session_state.file_uploader_key}")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if all(col in df.columns for col in feature_names):
                if st.button("🚀 Run Analysis & Generate Graphics", type="primary"):
                    batch_features = df[feature_names]
                    batch_scaled = scaler.transform(batch_features)
                    df['Prediction'] = ["Parkinson's" if p == 1 else "Healthy" for p in parkinsons_model.predict(batch_scaled)]
                    st.session_state.batch_results = df

                if 'batch_results' in st.session_state:
                    res = st.session_state.batch_results
                    st.divider()
                    
                    tab1, tab2, tab3 = st.tabs(["📈 Dynamic Comparison", "📋 Results Table", "🔥 Correlation Map"])
                    
                    with tab1:
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown("#### Axis Controls")
                            x_axis = st.selectbox("Select X-axis feature", feature_names, index=0)
                            y_axis = st.selectbox("Select Y-axis feature", feature_names, index=21) # Defaults to PPE
                            
                            # Small Pie Chart
                            fig_pie = px.pie(res, names='Prediction', hole=0.5,
                                           color_discrete_map={"Healthy": "#2ecc71", "Parkinson's": "#e74c3c"})
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with c2:
                            # DYNAMIC SCATTER PLOT
                            fig_scatter = px.scatter(res, x=x_axis, y=y_axis, color='Prediction',
                                                   title=f"Analysis: {x_axis} vs {y_axis}",
                                                   color_discrete_map={"Healthy": "#2ecc71", "Parkinson's": "#e74c3c"},
                                                   hover_data=feature_names[:3])
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                    with tab2:
                        st.dataframe(res, use_container_width=True)
                        st.download_button("📥 Download Report", res.to_csv(index=False), "parkinsons_report.csv")
                    
                    with tab3:
                        corr = res[feature_names].corr()
                        fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.error("Missing columns. Your CSV must have all 22 feature names.")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Clear Batch Data"):
        st.session_state.file_uploader_key += 1
        if 'batch_results' in st.session_state:
            del st.session_state.batch_results
        st.rerun()