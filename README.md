<img width="1348" height="584" alt="1_pred" src="https://github.com/user-attachments/assets/2dc5b8c9-5318-4ecb-87c5-d75eca5d7c21" />
<img width="1297" height="585" alt="2_pred" src="https://github.com/user-attachments/assets/704c91a5-8f01-4f55-a1e2-da42acbcf73d" />
<img width="1288" height="515" alt="3_pred" src="https://github.com/user-attachments/assets/89981515-2ef4-4b2e-b38f-81204dc8c96a" />
<img width="1348" height="584" alt="1_pred" src="https://github.com/user-attachments/assets/5508fe49-324d-4888-a80c-32d31a10111c" />
<img width="1297" height="585" alt="2_pred" src="https://github.com/user-attachments/assets/4868ec03-0572-4614-aa2f-d18879ff3dc7" />
<img width="1288" height="515" alt="3_pred" src="https://github.com/user-attachments/assets/26bc763b-2b4d-4224-8696-52301a818783" />
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
