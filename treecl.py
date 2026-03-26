import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

st.set_page_config(page_title="DendroClassifier", page_icon="🌲", layout="wide")

# ========== Load Data and Models ==========

@st.cache_data
def load_data():
    return pd.read_pickle('tree_data.pkl')

@st.cache_data
def load_labels():
    return pd.read_pickle('class_labels.pkl')

@st.cache_resource
def load_nn_models():
    scaler = joblib.load('scaler.joblib')
    nn_model = joblib.load('nn_model.joblib')
    return scaler, nn_model

@st.cache_resource
def load_cnn_model():
    return load_model("basic_cnn_tree_species.h5")

# ========== Utility Functions ==========

def recommend_species(input_data, nn_model, scaler, df, top_n=5):
    input_scaled = scaler.transform([input_data])
    distances, indices = nn_model.kneighbors(input_scaled)
    neighbors = df.iloc[indices[0]]
    species_counts = Counter(neighbors['common_name'])
    top_species = species_counts.most_common(top_n)
    return top_species

def get_common_locations_for_species(df, tree_name, top_n=10):
    species_df = df[df['common_name'] == tree_name]
    if species_df.empty:
        return pd.DataFrame(columns=['City', 'State', 'Recordings'])
    location_counts = species_df.groupby(['city', 'state']) \
                                .size().reset_index(name='count') \
                                .sort_values(by='count', ascending=False) \
                                .head(top_n)
    
    # Capitalize and format column names for the UI tables
    location_counts = location_counts.rename(columns={'city': 'City', 'state': 'State', 'count': 'Recordings'})
    return location_counts

# ========== Custom CSS ==========
st.markdown("""
<style>
    /* Premium App Background */
    .stApp {
        background-color: #F8FCF5;
    }
    
    /* Elegant Sidebar with Deep Green */
    [data-testid="stSidebar"] {
        background-color: #21381B;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ECF5E8 !important;
    }

    .reportview-container {
        background-color: transparent;
    }
    .big-title {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #5DA92F, #6EEE87);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #5FC52E;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }
    /* Style input boxes explicitly so they don't blend weirdly */
    .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: white !important;
        border: 1px solid #9BD46A !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== Main App ==========

def main():
    st.markdown('<p class="big-title">🌿 DendroClassifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI- Species Identification & Location Recommender</p>', unsafe_allow_html=True)
    st.divider()

    df = load_data()
    scaler, nn_model = load_nn_models()
    cnn_model = load_cnn_model()
    class_labels = load_labels()

    st.sidebar.markdown(
        """
        <div style="background-color: #9BD46A; padding: 8px 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: #1c3016; font-weight: bold; font-size: 1.4rem;">Navigation 🧭</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    mode = st.sidebar.radio("Select a Feature:", [
        "📷 Identify Tree from Image",
        "🌲 Recommend Trees by Location",
        "📍 Find Locations for a Tree"
    ])
    
    st.sidebar.divider()

    if mode == "🌲 Recommend Trees by Location":
        st.header("🌲 Recommend Trees by Location")
        st.markdown("Enter your geographic data below to find which trees are most likely to thrive in your area.")
        
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", -90.0, 90.0, 38.2274, format="%.6f", help="GPS Latitude")
            lon = st.number_input("Longitude", -180.0, 180.0, -85.8009, format="%.6f", help="GPS Longitude")
            diameter = st.number_input("Average Tree Diameter (cm)", 0.0, 1000.0, 1.0)
        with col2:
            native = st.selectbox("Native Status", df['native'].astype('category').cat.categories)
            city = st.selectbox("City", df['city'].astype('category').cat.categories)
            state = st.selectbox("State", df['state'].astype('category').cat.categories)

        native_code = df['native'].astype('category').cat.categories.get_loc(native)
        city_code = df['city'].astype('category').cat.categories.get_loc(city)
        state_code = df['state'].astype('category').cat.categories.get_loc(state)

        input_data = [lat, lon, diameter, native_code, city_code, state_code]

        if st.button("🔍 Get Recommendations", type="primary"):
            recommendations = recommend_species(input_data, nn_model, scaler, df, top_n=5)
            st.success("Analysis Complete! Here are your top recommendations:")
            
            for i, (species, count) in enumerate(recommendations, 1):
                st.info(f"**#{i}. {species.title()}**  \n*(Historically observed {count} times nearby)*")

    elif mode == "📍 Find Locations for a Tree":
        st.header("📍 Find Locations for a Tree")
        st.markdown("Wondering where a specific tree species grows best? Select it below!")
        
        tree_name = st.selectbox("Select Tree Species", sorted(df['common_name'].unique()))
        
        if st.button("🗺️ Show Common Locations", type="primary"):
            top_locations = get_common_locations_for_species(df, tree_name)
            if top_locations.empty:
                st.warning(f"No location data found for '{tree_name}'")
            else:
                st.success(f"Top Habitats for **{tree_name.title()}**:")
                # Use st.dataframe which natively handles hide_index without visual spacing glitches
                st.dataframe(top_locations, use_container_width=True, hide_index=True)

    elif mode == "📷 Identify Tree from Image":
        st.header("📷 Identify Tree from Image")
        st.markdown("Upload a clear photo of a leaf, bark, or full tree to instantly identify its species.")
        
        uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            image = Image.open(uploaded_file).convert('RGB')
            with col1:
                st.image(image, caption='Your Uploaded Subject', use_container_width=True)

            with st.spinner('Analyzing botanical features...'):
                IMG_SIZE = (224, 224)
                img = image.resize(IMG_SIZE)
                
                # We do NOT divide by 255.0 because the Rescaling layer is built natively into our Keras model
                img_array = img_to_array(img)  
                img_array = np.expand_dims(img_array, axis=0)

                predictions = cnn_model.predict(img_array)
                pred_idx = np.argmax(predictions)
                
                # Format the display label elegantly
                pred_label = class_labels[pred_idx]
                display_label = "Unclassified / Miscellaneous Species" if pred_label.lower() == "other" else pred_label.title()
                
                confidence = predictions[0][pred_idx]

            with col2:
                st.success("✅ **Identification Complete**")
                st.metric(label="Predicted Species", value=display_label)
                st.metric(label="AI Confidence", value=f"{confidence:.2%}")
                
                with st.expander("📊 View Top 3 Alternative Predictions"):
                    top_3_idx = predictions[0].argsort()[-3:][::-1]
                    for i in top_3_idx:
                        d_lbl = "Unclassified / Miscellaneous Species" if class_labels[i].lower() == "other" else class_labels[i].title()
                        st.write(f"- **{d_lbl}**: {predictions[0][i]:.2%}")

            st.divider()
            st.subheader(f"📌 Where does the **{display_label}** normally grow?")
            location_info = get_common_locations_for_species(df, pred_label)
            if location_info.empty:
                st.warning("This species is currently missing from our geolocation database.")
            else:
                st.dataframe(location_info, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
