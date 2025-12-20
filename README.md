
# 🌳 Tree Intelligence Assistant

This AI-powered app helps students and nature enthusiasts identify and explore tree species based on image, location, and tree attributes.

---

## 🧠 Features

<details>
<summary><strong>🌲 Recommend Trees by Location</strong></summary>

- Input GPS coordinates (latitude, longitude)
- Specify tree diameter, native status, city, and state
- Returns the top 5 tree species likely found in that area
</details>

<details>
<summary><strong>📍 Find Locations for a Tree</strong></summary>

- Choose a tree species from a dropdown
- Displays cities and states where the species is most commonly found
</details>

<details>
<summary><strong>📷 Identify Tree from Image</strong></summary>

- Upload an image of a tree
- CNN model predicts the species
- If found in the dataset, shows common locations for that species
</details>

---

## 📊 Dataset Description

<details>
<summary><strong>🗂️ Tree Metadata</strong></summary>

- **Source**: Open tree surveys from multiple cities (e.g., Louisville, Chicago)
- **Total records**: ~1.38 million
- **Key columns**:
  - `common_name`: Tree species (e.g., Bur Oak)
  - `scientific_name`: Botanical name (e.g., Quercus macrocarpa)
  - `latitude_coordinate`, `longitude_coordinate`
  - `city`, `state`, `address`
  - `native`: Whether the tree is native to the area
  - `diameter_breast_height_CM`: Tree height/width measure
</details>

<details>
<summary><strong>🖼️ Tree Image Dataset</strong></summary>

- **Structure**: Folder-based, each folder named after a tree species
- **Use**: Used to train the CNN for species recognition
- **Preprocessing**:
  - Images resized to 224x224
  - Normalized pixel values
  - Augmented with flips, zoom, and rotation
</details>

---

## 🧪 Algorithms Used

<details>
<summary><strong>🔍 Recommender System</strong></summary>

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Library**: `sklearn.neighbors.NearestNeighbors`
- **Inputs**: location, diameter, native status, city/state
- **Output**: Most common tree species nearby
</details>

<details>
<summary><strong>🧠 CNN Classifier</strong></summary>

- **Model**: Sequential CNN (Conv2D + MaxPooling + Dense layers)
- **Library**: `tensorflow.keras`
- **Input**: 224x224 image
- **Output**: Predicted tree species with probability
- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam
</details>

<details>
<summary><strong>📊 Preprocessing & Encoding</strong></summary>

- **Categorical Encoding**: LabelEncoder
- **Scaling**: StandardScaler for lat/lon/diameter
- **Data Splits**: 80% training, 20% validation
</details>

---

## ✅ How to Run

Run `tree.ipynb` to train the image classifier and save:
Launch the app:

streamlit run streamlit_integrated.py

 
