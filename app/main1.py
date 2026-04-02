import os
# import cv2
from insightface.app import FaceAnalysis
from utility import cosine_similarity, load_image, cv2

"""
ABOUT THIS SCRIPT:
This Script compares a query face against a group of faces and returns the best match. i.e. 1 → N.

i.e. 1 face (query) is compared against 1 image with multiple faces (group). The script identifies the best matching face in the group image and extracts it.

It uses the Greedy Matching approach, which means it looks for the best match for each query face independently. This is simpler and faster but can lead to suboptimal results if there are multiple similar faces in the group image.
"""

# configuration path 
SIMILARITY_THRESHOLD = 0.5
# OUTPUT_IMG = r"C:\Users\wings\sevenwings_inc\identity_classifier\test\output.jpg"
OUTPUT_IMG = os.path.join(os.getcwd(), "output.jpg")

# Initialize model (for CPU)
# app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)   # ctx_id=-1 for CPU, ctx_id=0 for GPU (if available)

# -------------------------------------
# Image processing function 
# -------------------------------------

def search_and_extract(group_img_path, query_img_path):
    # load image 
    group_img = load_image(group_img_path)
    query_img = load_image(query_img_path)

    # detect faces in the group image
    group_faces = app.get(group_img)
    query_faces = app.get(query_img)

    # Stop execution if no faces are found
    if not query_faces:
        raise ValueError("No face detected in query image.")
    if not group_faces:
        raise ValueError("No faces detected in group image.")
        
    # assuming there is a single face in the query image
    query_embedding = query_faces[0].embedding
    best_match = None
    best_score = -1

    # compare the query embedding with each face in the group image
    for face in group_faces:
        sim = cosine_similarity(query_embedding, face.embedding)
        if sim > best_score:
            best_score = sim
            best_match = face
        
    print(f"Best match similarity: {best_score:.4f}") 

    # Decision based on similarity threshold
    if best_score >= SIMILARITY_THRESHOLD:
        print("Match found. Extracting the face from the group image.")

        h, w, _ = group_img.shape        
        x1, y1, x2, y2 = best_match.bbox.astype(int)
        
        # Apply Clipping
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        extracted_face = group_img[y1:y2, x1:x2]
        cv2.imwrite(OUTPUT_IMG, extracted_face)

        print(f"Extracted face saved as {OUTPUT_IMG}")
        return OUTPUT_IMG
    else:
        print("No match found above the similarity threshold.")
        return None
    

# Run Program -- 
if __name__ == "__main__":
    group_image = r"C:\Users\wings\sevenwings_inc\identity_classifier\test\group1.jpg" 
    query_image = r"C:\Users\wings\sevenwings_inc\identity_classifier\test\person1.jpg"

    try:
        result = search_and_extract(group_image, query_image)
    except Exception as e:
        print(f"Error: {str(e)}")

        

