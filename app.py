import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- Page setup ---
st.set_page_config(page_title="Iceberg Detection", layout="centered")
st.title("🧊 Iceberg Detection in Maritime Radar Data")

# --- Load model safely ---
@st.cache_resource
def load_iceberg_model():
    try:
        model = load_model("iceberg_cnn.h5")
        return model
    except Exception as e:
        st.error("❌ Model file not found! Please make sure 'iceberg_cnn.h5' is in the same folder.")
        st.stop()

model = load_iceberg_model()

# --- File uploader ---
uploaded_file = st.file_uploader("Upload test data (CSV with columns: band_1, band_2, inc_angle)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        # Check for necessary columns
        required_cols = ['band_1', 'band_2', 'inc_angle']
        if not all(col in data.columns for col in required_cols):
            st.error(f"Missing required columns! Expected: {required_cols}")
        else:
            # Preprocess
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data[required_cols].values)

            # Make predictions
            preds = model.predict(X_scaled)
            data['Prediction'] = np.where(preds > 0.5, '🧊 Iceberg', '🚢 Ship')

            st.write("### Prediction Results")
            st.dataframe(data[['band_1', 'band_2', 'inc_angle', 'Prediction']])

            # Download option
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv, file_name="iceberg_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Error while processing file: {e}")
else:
    st.info("👆 Please upload a test CSV file to begin prediction.")

