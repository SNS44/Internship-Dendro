# 🌿 DendroClassifier

This application helps students and nature enthusiasts identify and explore tree species based on images, location data, and tree attributes. Let AI be your personal botanist!

---

## ✨ Key Highlights
- **Advanced CNN Engine**: Employs a robust image classifier utilizing Transfer Learning. Integrates strict "Class Penalties" to combat dataset imbalance, preventing the AI from lazily guessing generic classes. 
- **Elegant Edge-Case Handling**: Trees that the AI genuinely categorizes into the dataset's `other` folder are professionally displayed as **"Unclassified / Miscellaneous Species"**.

---

## 🧠 Core Features

<details>
<summary><strong>🌲 Recommend Trees by Location</strong></summary>

- Input GPS coordinates (latitude, longitude).
- Specify tree diameter, native status, city, and state.
- Returns the top 5 tree species mathematically most likely to thrive in your exact area using K-Nearest Neighbors.
</details>

<details>
<summary><strong>📍 Find Locations for a Tree</strong></summary>

- Choose a specific tree species from a dropdown menu.
- Instantly displays a cleanly formatted index-free table of the cities and states where that species is historically most commonly found.
</details>

<details>
<summary><strong>📷 Identify Tree from Image</strong></summary>

- Upload an image of a leaf, bark, or full tree.
- Our custom-trained CNN predicts the species and computes its confidence percentage via an attractive metric widget.
- Click to expand and view the Top 3 Alternative AI predictions.
- Instantly cross-references the prediction with the geolocation database to show you where the tree naturally grows!
</details>

---

## 📦 Requirements

- **Streamlit**: Web interface framework
- **TensorFlow & Keras**: Image classification
- **Scikit-Learn**: K-Nearest Neighbors recommendation engine
- **Pandas & NumPy**: Tabular data manipulation
- **Joblib**: Model serialization
- **Pillow (PIL)**: Image processing

---

## ✅ How to Run Locally

1. Open your terminal and navigate to your project folder:
   ```bash
   cd path/to/your/DendroClassifier
   ```
2. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit App!
   ```bash
   streamlit run treecl.py
   ```
