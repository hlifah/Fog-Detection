<img width="661" height="464" alt="b" src="https://github.com/user-attachments/assets/016f24a7-85bb-4f9a-af7b-6279a87b9a32" />
<img width="1297" height="585" alt="c" src="https://github.com/user-attachments/assets/dc7e0884-446a-4eb0-b9d9-5a3c81794a6c" />
<img width="1288" height="515" alt="d" src="https://github.com/user-attachments/assets/6c50bab9-b9f5-43c3-9e4b-b08ddc1a9403" />
<img width="1281" height="544" alt="a" src="https://github.com/user-attachments/assets/b31424bf-6cf6-4b72-82dd-de16a8c5c9ad" />

## 📖 Overview
In this project, we develop an advanced computer vision pipeline to classify atmospheric visibility (fog levels) using a state-of-the-art vision transformer model **BEiT** (Bidirectional Encoder representation from Image Transformers). 

---

## 🛠️ Project Pipeline Structure

1. **Problem Statement & Setup**: Importing required deep learning and utility libraries (`torch`, `transformers`, `torchvision`).
2. **Dataset Loading**: Setting paths and loading images categorized across varying fog densities (`Clear`, `Medium fog`, `Dense fog`).
3. **Data Preprocessing & Augmentation**: Configuring image sizing, normalization, and tensor conversions via Hugging Face processors.
4. **Model Building & Initialization**: Instantiating the pretrained `microsoft/beit-base-patch16-224` model checkpoint configured for custom classification labels.
5. **Training & Validation Engine**: Optimizing model parameters using `AdamW` and tracking cross-entropy loss across epochs.
6. **Interactive Streamlit Dashboard**: Deploying a real-time web interface for inference and live tactical safety advisories.

---

## 🚀 Getting Started & Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/fog-detection-beit.git](https://github.com/hlifah/fog-detection-beit.git)
cd fog-detection-beit
pip install -r requirements.txt
