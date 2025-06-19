from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

def st_file_to_pil(uploaded_file):
    return Image.open(uploaded_file).convert('RGB')

def pil_to_np(img):
    return np.array(img)

def draw_bboxes(image, bboxes, scores=None, color=(255,0,0), width=2):
    """
    image: PIL.Image
    bboxes: list of (x1, y1, x2, y2)
    scores: list of float or None
    Returns: PIL.Image with boxes drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    for i, box in enumerate(bboxes):
        draw.rectangle(box, outline=color, width=width)
        if scores is not None:
            text = f"{scores[i]:.2f}"
            draw.text((box[0], box[1]), text, fill=color, font=font)
    return img 