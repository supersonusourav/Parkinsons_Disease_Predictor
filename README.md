### Parlinsons Disease Predictor [App Link](https://parkinsonsdisease-predictor.streamlit.app/)

# Description: 
A production-ready health-tech application that leverages Gradient Boosting (XGBoost) to detect Parkinson’s Disease using vocal biomarker analysis. This system processes 22 distinct acoustic features (MDVP, Shimmer, Jitter, PPE) to provide real-time diagnostic probabilities.

# Key Features:

High-Accuracy ML Engine: Optimized XGBoost Classifier trained with GroupKFold cross-validation to prevent subject-based data leakage.

Dynamic Batch Analytics: Upload entire CSV datasets to generate instant health reports with interactive Plotly visualizations (Scatter Plots & Correlation Heatmaps).

Smart Input System: Features a "Quick Fill" CSV parser that instantly maps raw data strings to individual feature metrics.

Clinician-Centric UI: Built with Streamlit, providing confidence scores and diagnostic breakdowns in a clean, responsive interface.

Tech Stack: Python | XGBoost | Streamlit | Scikit-Learn | Plotly | Pandas | Pickle
