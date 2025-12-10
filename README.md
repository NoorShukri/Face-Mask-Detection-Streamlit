# 🟢 Face Mask Detection

Date: December 10, 2025
Author: Noor Shukri
Supervisor: Sarah Abdelsalam

-----

## 1️⃣ Problem Definition
During the COVID-19 pandemic, face masks became one of the most important preventive measures to reduce virus transmission.  
Monitoring mask compliance manually in crowded areas is difficult and inefficient.  

**Goal:** Build an intelligent system that automatically detects whether a person is wearing a face mask using Computer Vision and Deep Learning.  

**Supported Inputs:**  
- Static images  
- Video files  
- Real-time webcam streams  

**Interface:** Interactive **Streamlit application**.

---

## 2️⃣ Dataset
The project uses a Mask / No-Mask dataset with variations in:  
- ✅ Lighting conditions  
- ✅ Face angles  
- ✅ Backgrounds  
- ✅ Age and gender  

**Dataset Splitting:**  
- 70% Training  
- 20% Validation  
- 10% Testing  

**Data Preparation:**  
- Balanced classes for Mask / No Mask  
- Converted all images to **224×224** resolution  
- Organized in structured folders  
- Efficient preprocessing pipelines

---

## 3️⃣ Preprocessing
Before training, the following steps were applied:  

- **Image Resizing:** All images resized to 224×224  
- **Normalization:** Pixel values scaled to [0,1]  
- **Data Augmentation:** Rotation, Zoom, Horizontal Flip, Brightness changes  
- **Face Detection in App:** MediaPipe used to crop faces before classification

---

## 4️⃣ Feature Extraction & Algorithm
**Model Architecture:** MobileNetV2  
- Lightweight and fast  
- High accuracy with low computational cost  
- Uses Depthwise Separable Convolutions  

**Transfer Learning Steps:**  
- Freeze base MobileNetV2 layers  
- Add fully-connected layers for Mask/No-Mask classification  
- Train classification head  
- Fine-tune selected layers  

**Output:** Two probabilities → Mask or No Mask (prediction based on higher probability)

---

## 5️⃣ Model Training
**Configuration:**  
- Optimizer: Adam  
- Loss Function: Binary Crossentropy  
- Batch Size: Tuned experimentally  
- Epochs: Early Stopping  
- Learning Rate: Adjusted for stable convergence  

**Callbacks Used:**  
- EarlyStopping  
- ModelCheckpoint  
- ReduceLROnPlateau  

**Fine-Tuning:** Unfreeze some MobileNetV2 layers after initial training to improve accuracy

---

## 6️⃣ Evaluation
**Metrics:**  
- Accuracy  
- Loss  
- Precision  
- Recall  
- Confusion Matrix  

**Observations:**  
- High accuracy distinguishing Mask vs. No Mask  
- Works well under different lighting and angles  
- Small or distant faces are challenging, but MediaPipe improves detection

---

## 7️⃣ System Integration (Streamlit App)
**Features:**  
- **Image Input:** Upload → detect face → crop → classify → bounding box & confidence  
- **Video Input:** Frame-by-frame mask detection  
- **Live Webcam Mode:** Real-time detection  

**Face Detection:** MediaPipe for fast & accurate bounding boxes  
**Visual Output:**  
- 🟢 Green box: Mask  
- 🔴 Red box: No Mask  
- Confidence score displayed near the box

---

## 8️⃣ Project Structure 
The repository is organized as follows:

final_cv/
├── cv_project.py             # The main Streamlit application script
├── final_best_mask_model.h5  # The trained MobileNetV2 model
├── mask_model_cv.ipynb       # Jupyter Notebook used for training
├── requirements.txt          # Python dependencies
└── data/                     # Dataset folder (used for training)

---

## 9️⃣ Installation & Usage
Step 1 - Clone the repository to run the project locally:
git clone https://github.com/NoorShukri/Face-Mask-Detection-Streamlit.git
cd Face-Mask-Detection-Streamlit

Step 2 - Install dependencies
pip install -r requirements.txt

Step 3 - Run the Streamlit app
streamlit run cv_project.py

## 🔟 Conclusion
**Highlights:**  
- Lightweight & fast model  
- Real-time performance  
- High accuracy  
- User-friendly interface  

**Use Cases:**  
- Universities  
- Public transportation  
- Hospitals  
- Shopping malls  
- Workplaces  

**Future Improvements:**  
- Add class for incorrect mask wearing  
- Improve performance on very small/distant faces  
- Build a monitoring dashboard with logs/statistics  
- Deploy as cloud-based surveillance tool

---

## 1️⃣1️⃣ References
- TensorFlow Documentation  
- MediaPipe Face Detection  
- MobileNetV2 Research Paper  
- ImageNet Dataset

