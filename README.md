<img width="661" height="464" alt="images" src="https://github.com/user-attachments/assets/9ef1d827-4b31-4b17-84a8-c1e5f83b4495" />
<img width="1297" height="585" alt="2_pred" src="https://github.com/user-attachments/assets/037f4b09-0712-4874-b35b-72ccd63e25e1" />
<img width="1288" height="515" alt="3_pred" src="https://github.com/user-attachments/assets/bfc78f91-67aa-4814-a089-055afd832743" />
<img width="1281" height="544" alt="1_pred" src="https://github.com/user-attachments/assets/41b80a10-9c9e-4960-94a6-c0e2d41337d6" />
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
