# Parlinsons Disease Predictor [App Link](https://parkinsonsdisease-predictor.streamlit.app/)

<img width="1867" height="808" alt="image" src="https://github.com/user-attachments/assets/8b18796e-797e-403b-84b2-61307aa9e94c" />
<img width="1815" height="824" alt="image" src="https://github.com/user-attachments/assets/7e2d9641-de6e-4713-937b-bd91db66d60a" />


## Description: 
A production-ready health-tech application that leverages Gradient Boosting (XGBoost) to detect Parkinson’s Disease using vocal biomarker analysis. This system processes 22 distinct acoustic features (MDVP, Shimmer, Jitter, PPE) to provide real-time diagnostic probabilities.

## Key Features:

High-Accuracy ML Engine: Optimized XGBoost Classifier trained with GroupKFold cross-validation to prevent subject-based data leakage.

Dynamic Batch Analytics: Upload entire CSV datasets to generate instant health reports with interactive Plotly visualizations (Scatter Plots & Correlation Heatmaps).

Smart Input System: Features a "Quick Fill" CSV parser that instantly maps raw data strings to individual feature metrics.

Clinician-Centric UI: Built with Streamlit, providing confidence scores and diagnostic breakdowns in a clean, responsive interface.

Tech Stack: Python | XGBoost | Streamlit | Scikit-Learn | Plotly | Pandas | Pickle
