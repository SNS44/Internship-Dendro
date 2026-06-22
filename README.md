# Dendro

Project: Shell-Edunet Skills4Future AICTE Internship — Green Skills | AI & Data Analytics

<h1 align="center">🌿 DENDRO</h1>

<p align="center">
  <b><i>Identify trees. Discover habitats. Explore nature — powered by AI.</i></b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Demo%20Project-4CAF50?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Focus-Green%20Skills%20%26%20AI-2E7D32?style=for-the-badge" alt="Focus">
  <img src="https://img.shields.io/badge/Program-Shell--Edunet%20AICTE-F4B400?style=for-the-badge" alt="Program">
</p>

<div align="center">

[![Python](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Data-Pandas%20%26%20NumPy-150458?style=flat-square&logo=pandas&logoColor=white)](#)

</div>

---

## 🌱 The Vision

**Dendro** is an AI-powered botanical web application designed to make tree identification and nature exploration accessible to everyone — from students to environmentalists. It uses deep learning and geospatial data to instantly name a tree species from a photo, recommend the best trees for any location, and show exactly where a species thrives in the world.

This project was developed for the **Shell-Edunet Skills4Future AICTE Internship**, a program presented by the **Edunet Foundation** in collaboration with **AICTE & Shell**, focusing on **Green Skills using AI technologies**. It provides participants the opportunity to build real projects under expert industry mentorship, entirely at no cost.


---

## 🚀 Features

* 📷 **Identify Tree from Image** — Upload a photo of a leaf, bark, or full tree and the AI names the species instantly
* 🌲 **Recommend Trees by Location** — Enter GPS coordinates and local details to find the top 5 trees most likely to thrive in your area
* 📍 **Find Locations for a Tree** — Pick any species and see the top 10 cities and states where it historically grows most
* 🔍 **Automatic Habitat Lookup** — After identifying a tree from a photo, the app shows where it grows — no extra steps
* 🧠 **Top 3 Alternatives** — View the next best predictions for any image identification
* 🧊 **Premium UI** — Dark green sidebar, gradient title, and clean layout built with Streamlit and custom CSS

---

## 🧠 How It Works (Core Logic)

**Image Identification:**

1. Upload a photo (JPG or PNG)
2. Image is resized to 224×224 and converted to a numeric array
3. The CNN model processes the array and outputs a confidence score for every known species
4. The species with the highest score is shown as the prediction
5. If the result is categorised as "other", the app displays it as **"Unclassified / Miscellaneous Species"**
6. The geolocation dataset is automatically queried to show where that species is most commonly found

**Location-Based Recommendation:**

```
Input: Latitude, Longitude, Tree Diameter, Native Status, City, State
         ↓
Categorical Encoding (same system model was fitted on)
         ↓
StandardScaler normalises the input vector
         ↓
KNN computes distances to all historical tree observations
         ↓
Top 5 most frequent species among nearest neighbors → Recommendations
```

---

## 🖥️ System Requirements

| Requirement | Details |
| ----------- | ------- |
| OS | Windows / macOS / Linux |
| Python | 3.9 – 3.11 (3.10 recommended) |
| Hardware | Standard computer — no GPU required |

---

## 📦 Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

Key packages:

| Library | Purpose |
| ------- | ------- |
| streamlit | Web interface |
| tensorflow | CNN image recognition |
| scikit-learn | KNN recommendation + scaling |
| pandas | Dataset queries and data handling |
| numpy | Numeric operations |
| joblib | Loading saved model files |
| Pillow | Image loading and preprocessing |

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone https://github.com/SNS44/Internship-Dendro.git
cd Internship-Dendro
```

2. (Recommended) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Launch the app

```bash
streamlit run treecl.py
```

The app will automatically launch and open in your web browser via the Streamlit interface.

---

## 🎮 Usage Instructions

1. Open the app — the **dark green sidebar** shows three navigation options
2. **Identify a tree from a photo** → Select *"📷 Identify Tree from Image"*, upload your image, wait for analysis
3. **Get tree recommendations** → Select *"🌲 Recommend Trees by Location"*, fill in coordinates and area details, click *"🔍 Get Recommendations"*
4. **Find where a species grows** → Select *"📍 Find Locations for a Tree"*, pick a species, click *"🗺️ Show Common Locations"*

---

## ⚙️ Configuration Parameters

You can adjust these defaults in `treecl.py` to tune the app:

```python
TOP_N_RECOMMENDATIONS = 5    # Number of species returned by KNN
TOP_N_LOCATIONS = 10         # Number of habitat locations shown per species
IMG_SIZE = (224, 224)        # CNN input image resolution
DEFAULT_LAT = 38.2274        # Default latitude
DEFAULT_LON = -85.8009       # Default longitude
```

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────┐
│              Streamlit UI Layer             │
│       (Sidebar Navigation + Custom CSS)     │
└──────┬───────────────┬──────────────────────┘
       │               │               │
       ▼               ▼               ▼
┌────────────┐  ┌──────────────┐  ┌──────────────┐
│  Upload    │  │   Location   │  │   Species    │
│  an Image  │  │  Input Form  │  │   Selector   │
└─────┬──────┘  └──────┬───────┘  └──────┬───────┘
      │                │                  │
      ▼                ▼                  ▼
┌────────────┐  ┌──────────────┐  ┌──────────────┐
│ TensorFlow │  │ StandardScaler│  │ Pandas Query │
│ CNN (.h5)  │  │  + KNN Model │  │  on Dataset  │
└─────┬──────┘  └──────┬───────┘  └──────┬───────┘
      │                │                  │
      ▼                ▼                  ▼
┌────────────┐  ┌──────────────┐  ┌──────────────┐
│ Species +  │  │  Top 5 Tree  │  │  Top 10      │
│ Confidence │  │ Recommenda-  │  │  Habitat     │
│ + Top 3    │  │   tions      │  │  Locations   │
└─────┬──────┘  └──────────────┘  └──────────────┘
      │
      ▼
┌────────────┐
│ Auto Runs  │
│  Habitat   │
│   Lookup   │
└────────────┘
```

### Architectural Design Choices

* **Cached models** — CNN and KNN are loaded once into memory so every request is instant
* **Rescaling layer inside CNN** — images don't need manual normalisation before inference, it's handled inside the model
* **Category encoding from master list** — uses the exact same encoding order the scaler was trained on, preventing silent data mismatches
* **Automatic habitat cross-reference** — runs immediately after image prediction without any extra user action

---

## 🔁 Processing Flowchart

### Image Identification Flow

```
START
  │
  ▼
User Uploads Image
  │
  ▼
Resize → 224×224 + Convert to Array
  │
  ▼
CNN Model Predicts Probabilities
  │
  ▼
Top Confidence ≥ "other" threshold?
  ├── YES → Show Species Name + Confidence
  │
  └── NO  → Show "Unclassified / Miscellaneous Species"

  │
  ▼
Auto Query Habitat Dataset
  │
  ▼
Display Top 10 Locations for Predicted Species
  │
  ▼
END
```

---

## 🧪 Known Limitations

* CNN only recognises species it was trained on — unusual trees may be labelled as "Unclassified"
* Blurry, dark, or cluttered photos reduce prediction accuracy
* Location recommendations are based on historical observations, not live climate data
* Dataset covers mainly US cities and states — results outside that region may be limited

---

## 🛠️ Tech Stack

* **Python** — Core language
* **Google Colab** — Model training environment (CNN trained on cloud GPU)
* **Streamlit** — Web UI and app framework
* **TensorFlow / Keras** — CNN image classification (Transfer Learning)
* **Scikit-Learn** — K-Nearest Neighbors recommendation engine + StandardScaler
* **Pandas** — Dataset querying and geolocation table rendering
* **NumPy** — Array operations and numeric processing
* **Joblib** — Loading serialised model files (.joblib)
* **Pillow (PIL)** — Image opening and preprocessing

---

## 👤 Developer

**Shreyas**

Submitted for the **Shell-Edunet Skills4Future AICTE Internship** — a program by the Edunet Foundation in collaboration with AICTE & Shell, designed to build Green Skills and AI competencies under real industry mentorship.

---

## ✅ Evaluation Readiness

✔ Real AI model — CNN trained on actual tree species image data  
✔ Multi-modal input — handles both image uploads and structured data  
✔ End-to-end pipeline — raw user input all the way to a meaningful result  
✔ Green Skills focus — built around tree biodiversity and conservation awareness  
✔ Production-style UI — clean design, smooth navigation, proper error handling  

This is not a toy project. It demonstrates how AI can make environmental and botanical knowledge practical and accessible for everyone.


---

<p align="center">
  <b>Built with 🌿 for a Greener Future</b>
  <br/>
  <i>Shell-Edunet Skills4Future · AICTE · Edunet Foundation</i>
</p>
