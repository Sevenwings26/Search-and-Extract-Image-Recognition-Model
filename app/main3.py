import os
import shutil
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from utility import load_image

# --- Configuration ---
SIMILARITY_THRESHOLD = 0.55  # Slightly higher for event sorting to ensure accuracy
GALLERY_DIR = r"C:/Users/wings/Desktop/7wingsINC_dev/neoEvents/app/identity_classifier/event_images" # The source folder
USER_STORAGE_DIR = "attendee_folders" # The destination root

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-6)

def search_and_sort_event(query_img_path, attendee_name):
    """
    Finds all event images containing the attendee and copies them 
    to a personalized folder.
    """
    # 1. Setup personalized folder
    attendee_dir = os.path.join(USER_STORAGE_DIR, attendee_name)
    os.makedirs(attendee_dir, exist_ok=True)

    # 2. Get Query Embedding (The Attendee's "Key")
    query_img = load_image(query_img_path)
    query_faces = app.get(query_img)
    if not query_faces:
        raise ValueError("Could not detect your face in the query image.")
    
    q_emb = normalize(query_faces[0].embedding)
    found_count = 0

    # 3. Iterate through the Event Gallery
    print(f"Scanning gallery for {attendee_name}...")
    for filename in os.listdir(GALLERY_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        file_path = os.path.join(GALLERY_DIR, filename)
        event_img = cv2.imread(file_path)
        
        if event_img is None: continue

        # Detect all faces in the event photo
        faces_in_photo = app.get(event_img)
        
        is_match = False
        for face in faces_in_photo:
            g_emb = normalize(face.embedding)
            similarity = np.dot(q_emb, g_emb)

            if similarity >= SIMILARITY_THRESHOLD:
                is_match = True
                break # Stop checking this photo once the person is found

        # 4. If person is in photo, copy the FULL image
        if is_match:
            shutil.copy(file_path, os.path.join(attendee_dir, filename))
            found_count += 1
            print(f"Match found in: {filename} (Score: {similarity:.4f})")

    return found_count

if __name__ == "__main__":
    # Example usage
    # query = r"C:\path\to\attendee_selfie.jpg"
    query = r"C:/Users/wings/Desktop/7wingsINC_dev/neoEvents/app/identity_classifier/query/person1.jpg"
    try:
        count = search_and_sort_event(query, "John_Doe")
        print(f"Finished! Found {count} images for the attendee.")
    except Exception as e:
        print(f"Error: {str(e)}")

