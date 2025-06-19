import streamlit as st
from PIL import Image
import numpy as np
import io

# Try to import st_canvas for annotation
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

from backend.utils import st_file_to_pil, draw_bboxes, pil_to_np
from backend.embedding import load_dinov2_model, extract_embedding
from backend.graph import build_object_graph, find_major_object
from backend.detection import sliding_window_candidates, compare_embeddings, non_max_suppression

st.set_page_config(page_title="Pattern-Based Object Detection Prototype", layout="wide")
st.title("Pattern-Based Object Detection Prototype for Aerial Reconnaissance")

st.sidebar.header("Step 1: Upload Images")
pattern_img = st.sidebar.file_uploader("Upload Pattern Image", type=["jpg", "jpeg", "png"])
query_img = st.sidebar.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])

st.header("Step 2: Annotate Pattern Image")

bboxes = []
pattern_pil_img = None
if pattern_img:
    pattern_pil_img = st_file_to_pil(pattern_img)
    if 'bboxes' not in st.session_state:
        st.session_state['bboxes'] = []
    if CANVAS_AVAILABLE:
        st.write("Draw bounding boxes on the pattern image below:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=pattern_pil_img,
            update_streamlit=True,
            height=pattern_pil_img.height,
            width=pattern_pil_img.width,
            drawing_mode="rect",
            key="canvas",
        )
        if canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                left = obj["left"]
                top = obj["top"]
                width = obj["width"]
                height = obj["height"]
                x1, y1, x2, y2 = int(left), int(top), int(left+width), int(top+height)
                bboxes.append((x1, y1, x2, y2))
            st.session_state['bboxes'] = bboxes
        if bboxes:
            st.image(draw_bboxes(pattern_pil_img, bboxes), caption="Annotated Pattern Image", use_column_width=True)
    else:
        st.warning("streamlit-drawable-canvas not installed. Please enter bounding boxes manually.")
        if st.button("Add Bounding Box"):
            st.session_state['bboxes'].append((0,0,100,100))
        for i, box in enumerate(st.session_state['bboxes']):
            x1 = st.number_input(f"Box {i+1} - x1", value=box[0], key=f"x1_{i}")
            y1 = st.number_input(f"Box {i+1} - y1", value=box[1], key=f"y1_{i}")
            x2 = st.number_input(f"Box {i+1} - x2", value=box[2], key=f"x2_{i}")
            y2 = st.number_input(f"Box {i+1} - y2", value=box[3], key=f"y2_{i}")
            st.session_state['bboxes'][i] = (x1, y1, x2, y2)
        bboxes = st.session_state['bboxes']
        if bboxes:
            st.image(draw_bboxes(pattern_pil_img, bboxes), caption="Annotated Pattern Image", use_column_width=True)
else:
    st.info("Please upload a pattern image to annotate.")

st.header("Step 3: Major Object Identification")
major_idx = None
major_embedding = None
if pattern_img and bboxes:
    with st.spinner("Loading DINOv2 and extracting embeddings..."):
        model, device = load_dinov2_model()
        pattern_embeddings = []
        for box in bboxes:
            crop = pattern_pil_img.crop(box)
            emb = extract_embedding(model, crop, device)
            pattern_embeddings.append(emb)
        G = build_object_graph(bboxes)
        major_idx = find_major_object(G, method='centrality')
        major_embedding = pattern_embeddings[major_idx]
        st.success(f"Major object identified: Box #{major_idx+1}")
        st.image(draw_bboxes(pattern_pil_img, [bboxes[major_idx]], color=(0,255,0)), caption="Major Object Highlighted", use_column_width=True)
else:
    st.info("Please annotate at least one object to identify the major object.")

st.header("Step 4: Similar Object Detection in Query Image")
if query_img and major_embedding is not None:
    query_pil_img = st_file_to_pil(query_img)
    query_np_img = pil_to_np(query_pil_img)
    st.subheader("Detection Parameters")
    window_size = st.slider("Sliding Window Size", 32, 256, 64, 16)
    stride = st.slider("Sliding Window Stride", 8, 128, 32, 8)
    sim_thresh = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.01)
    nms_thresh = st.slider("NMS IoU Threshold", 0.1, 0.9, 0.5, 0.01)
    if st.button("Run Detection"):
        with st.spinner("Extracting candidate regions and computing similarities..."):
            candidates = sliding_window_candidates(query_np_img, window_size=window_size, stride=stride)
            candidate_embeddings = []
            for box in candidates:
                crop = query_pil_img.crop(box)
                emb = extract_embedding(model, crop, device)
                candidate_embeddings.append(emb)
            candidate_embeddings = np.stack(candidate_embeddings)
            sims = compare_embeddings(major_embedding, candidate_embeddings)
            # Filter by threshold
            keep_idx = np.where(sims >= sim_thresh)[0]
            keep_boxes = [candidates[i] for i in keep_idx]
            keep_scores = [float(sims[i]) for i in keep_idx]
            # NMS
            nms_idx = non_max_suppression(keep_boxes, keep_scores, iou_thresh=nms_thresh)
            final_boxes = [keep_boxes[i] for i in nms_idx]
            final_scores = [keep_scores[i] for i in nms_idx]
            st.success(f"Detected {len(final_boxes)} objects similar to the major pattern.")
            st.image(draw_bboxes(query_pil_img, final_boxes, final_scores), caption="Detection Results", use_column_width=True)
else:
    st.info("Please upload a query image and annotate/select a major object to run detection.") 