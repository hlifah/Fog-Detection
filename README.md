<img width="1281" height="544" alt="a" src="https://github.com/user-attachments/assets/dd0cb709-308f-4723-af09-1d15b2c88f8a" />
<img width="661" height="464" alt="b" src="https://github.com/user-attachments/assets/8d600647-89f7-4d16-8d04-087ff4c581ad" />
<img width="1297" height="585" alt="c" src="https://github.com/user-attachments/assets/1ae68fe1-50ef-4183-8969-44e724722828" />
<img width="1288" height="515" alt="d" src="https://github.com/user-attachments/assets/0599f644-dc0d-4fef-8636-f04ffbdc9a39" />



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
