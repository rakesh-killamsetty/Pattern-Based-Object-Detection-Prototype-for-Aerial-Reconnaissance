import numpy as np
# from skimage.util import view_as_windows
from sklearn.metrics.pairwise import cosine_similarity

def sliding_window_candidates(image, window_size=64, stride=32):
    """
    image: np.ndarray (H, W, C)
    Returns: list of (x1, y1, x2, y2) candidate boxes
    """
    H, W = image.shape[:2]
    boxes = []
    for y in range(0, H-window_size+1, stride):
        for x in range(0, W-window_size+1, stride):
            boxes.append((x, y, x+window_size, y+window_size))
    return boxes

def compare_embeddings(query_emb, candidate_embs):
    """
    query_emb: np.ndarray (D,)
    candidate_embs: np.ndarray (N, D)
    Returns: similarity scores (N,)
    """
    return cosine_similarity([query_emb], candidate_embs)[0]

def non_max_suppression(boxes, scores, iou_thresh=0.5):
    """
    boxes: list of (x1, y1, x2, y2)
    scores: list of float
    Returns: indices of selected boxes
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        inter = w*h
        ovr = inter/(areas[i]+areas[order[1:]]-inter)
        inds = np.where(ovr<=iou_thresh)[0]
        order = order[inds+1]
    return keep 