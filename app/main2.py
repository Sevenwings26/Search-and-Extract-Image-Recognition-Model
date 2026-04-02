import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.optimize import linear_sum_assignment # Hungarian Algorithm for optimal matching to reduce false positives and improve cost calculation, and assignment of query faces to group faces.
from utility import load_image


"""
ABOUT THIS SCRIPT:
This script performs Multi-Identity Resolution (N → N). 

It utilizes a 'Global Best-Fit' logic (inspired by the Hungarian Algorithm) 
to match multiple query individuals against a group photo. 

Unlike 'Greedy' matching—where the first query simply grabs the best 
available face—this approach ensures the most accurate pairing for the 
entire group. It prevents 'Identity Theft' (where two queries claim the 
same person) and optimizes the total confidence score of all matches combined.
"""

"""
Possible extensions:
1. Visualization: Draw bounding boxes + labels on group image
2. API Layer: Upload → return JSON + cropped faces
3. Tracking System: Extend to video (frame-by-frame identity consistency)
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

    # -----------------------------
    # STEP 1: Build similarity matrix
    # -----------------------------
    similarity_matrix = np.zeros((len(query_faces), len(group_faces)))

    for i, q_face in enumerate(query_faces):
        q_emb = normalize(q_face.embedding)

        for j, g_face in enumerate(group_faces):
            g_emb = normalize(g_face.embedding)
            similarity_matrix[i, j] = cosine_similarity(q_emb, g_emb)

    # -----------------------------
    # STEP 2: Convert to cost matrix
    # -----------------------------
    cost_matrix = 1 - similarity_matrix

    # -----------------------------
    # STEP 3: Hungarian Assignment --- Hungarian Algorithm looks at the scores for everyone and decides the pairing that makes the most sense for the group as a whole, rather than just picking the best match for each query independently.
    # -----------------------------
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results = []

    # -----------------------------
    # STEP 4: Process matches
    # -----------------------------
    for i, j in zip(row_ind, col_ind):
        score = similarity_matrix[i, j]

        print(f"[Query {i}] matched with Group {j} | similarity: {score:.4f}")

        if score < SIMILARITY_THRESHOLD:
            print(f"[Query {i}] Match below threshold.")
            results.append(None)
            continue

        matched_face = group_faces[j]

        # Expand bbox
        x1, y1, x2, y2 = expand_bbox(
            matched_face.bbox, group_img.shape, MARGIN
        )

        crop = group_img[y1:y2, x1:x2]

        if crop.size == 0:
            results.append(None)
            continue

        output_path = os.path.join(OUTPUT_DIR, f"person_{i}.jpg")
        cv2.imwrite(output_path, crop)

        results.append({
            "path": output_path,
            "confidence": float(score),
            "group_face_index": int(j)
        })

    return results


# run Program --
if __name__ == "__main__":
    group_image = r"C:\Users\wings\sevenwings_inc\identity_classifier\test\group1.jpg" 
    query_image = r"C:\Users\wings\sevenwings_inc\identity_classifier\test\person1.jpg"

    try:
        results = search_and_extract_multiple(group_image, query_image)
        print("Results:", results)
    except Exception as e:
        print(f"Error: {str(e)}")

























# def search_and_extract_multiple(group_img_path, query_img_path):
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     group_img = load_image(group_img_path)
#     query_img = load_image(query_img_path)

#     group_faces = app.get(group_img)
#     query_faces = app.get(query_img)

#     if not query_faces:
#         raise ValueError("No faces in query image.")
#     if not group_faces:
#         raise ValueError("No faces in group image.")

#     # Normalize group embeddings once
#     group_embeddings = [
#         (idx, face, normalize(face.embedding)) for idx, face in enumerate(group_faces)
#     ]

#     used_faces = set()
    
#     results = []

#     for i, q_face in enumerate(query_faces):
#         q_emb = normalize(q_face.embedding)

#         best_match = None
#         best_score = -1
#         best_idx = None

#         for idx, face, g_emb in group_embeddings:
#             # 🚫 Skip already used faces
#             if idx in used_faces:
#                 continue

#             sim = cosine_similarity(q_emb, g_emb)
#             if sim > best_score:
#                 best_score = sim
#                 best_match = face
#                 best_idx = idx

#         print(f"[Query {i}] Best similarity: {best_score:.4f}")

#         if best_score < SIMILARITY_THRESHOLD:
#             print(f"[Query {i}] No match found.")
#             results.append(None)
#             continue

#         # ✅ Mark this face as used
#         used_faces.add(best_idx)

#         # Extract face
#         x1, y1, x2, y2 = expand_bbox(best_match.bbox, group_img.shape, MARGIN)
#         crop = group_img[y1:y2, x1:x2]

#         if crop.size == 0:
#             results.append(None)
#             continue

#         output_path = os.path.join(OUTPUT_DIR, f"person_{i}.jpg")
#         cv2.imwrite(output_path, crop)

#         results.append({
#             "path": output_path,
#             "confidence": float(best_score)
#         })

#     return results


