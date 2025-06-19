# Pattern-Based Object Detection Prototype for Aerial Reconnaissance

## Overview
This project is a prototype system for pattern-based object detection in aerial imagery. It enables users to:
- Upload a pattern image and a query image
- Annotate objects of interest in the pattern image
- Identify the major object in the pattern
- Detect and highlight similar objects in the query image using DINOv2 embeddings and similarity search
- Visualize results with bounding boxes and confidence scores

## Features
- Streamlit-based interactive UI for image upload, annotation, and visualization
- DINOv2 (via PyTorch Hub) for object-level feature extraction
- Graph-based analysis (NetworkX) for major object identification
- Similarity-based detection using cosine similarity
- Non-maximum suppression for final detection

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Download sample aerial images** and place them in the `sample_data/` directory.

## Running the App
```bash
streamlit run app.py
```

## Project Structure
```
pattern-object-detection/
│
├── app.py                # Streamlit app (UI + pipeline)
├── backend/
│   ├── embedding.py      # Embedding extraction logic
│   ├── graph.py          # Graph construction and major object identification
│   ├── detection.py      # Candidate extraction, similarity, NMS
│   └── utils.py          # Helper functions
├── requirements.txt
├── README.md
└── sample_data/
```

## References
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NetworkX Documentation](https://networkx.org/)

---
**Note:** This is a prototype for research and demonstration purposes. For real-world deployment, further optimization and validation are required. 