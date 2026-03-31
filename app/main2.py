import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .utility import load_image


"""
ABOUT THIS SCRIPT:
This Script compares multiple query faces against a group of faces and returns the best match for each query. i.e. N → N...
"""


# configuration path 
SIMILARITY_THRESHOLD = 0.5
OUTPUT_DIR = "outputs"
MARGIN = 0.2

# Initialize model (for CPU)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)   # ctx_id=-1 for CPU, ctx_id=0 for GPU (if available)


def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-6)


def cosine_similarity(a, b):
    return np.dot(a, b)


def expand_bbox(bbox, img_shape, margin=0.2):
    h, w, _ = img_shape
    x1, y1, x2, y2 = bbox.astype(int)

    width = x2 - x1
    height = y2 - y1

    x1 -= int(width * margin)
    y1 -= int(height * margin)
    x2 += int(width * margin)
    y2 += int(height * margin)

    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def search_and_extract_multiple(group_img_path, query_img_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    group_img = load_image(group_img_path)
    query_img = load_image(query_img_path)

    group_faces = app.get(group_img)
    query_faces = app.get(query_img)

    if not query_faces:
        raise ValueError("No faces in query image.")
    if not group_faces:
        raise ValueError("No faces in group image.")

    # Normalize group embeddings once
    group_embeddings = [
        (face, normalize(face.embedding)) for face in group_faces
    ]

    used_faces = set()
    
    results = []

    for i, q_face in enumerate(query_faces):
        q_emb = normalize(q_face.embedding)

        best_match = None
        best_score = -1
        best_idx = None

        for idx, face, g_emb in group_embeddings:
            # 🚫 Skip already used faces
            if idx in used_faces:
                continue

            sim = cosine_similarity(q_emb, g_emb)
            if sim > best_score:
                best_score = sim
                best_match = face

        print(f"[Query {i}] Best similarity: {best_score:.4f}")

        if best_score < SIMILARITY_THRESHOLD:
            print(f"[Query {i}] No match found.")
            results.append(None)
            continue

        # ✅ Mark this face as used
        used_faces.add(best_idx)

        # Extract face
        x1, y1, x2, y2 = expand_bbox(best_match.bbox, group_img.shape, MARGIN)
        crop = group_img[y1:y2, x1:x2]

        if crop.size == 0:
            results.append(None)
            continue

        output_path = os.path.join(OUTPUT_DIR, f"person_{i}.jpg")
        cv2.imwrite(output_path, crop)

        results.append({
            "path": output_path,
            "confidence": float(best_score)
        })

    return results
