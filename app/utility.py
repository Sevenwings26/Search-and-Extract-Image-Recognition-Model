import numpy as np
import cv2


# utility function to calculate cosine similarity and load image

# What is cosine similarity? It is a measure of similarity between two non-zero vectors in an inner product space. It is defined as the cosine of the angle between them, which ranges from -1 to 1. A value of 1 indicates that the vectors are identical, while a value of -1 indicates that they are completely opposite. A value of 0 indicates that the vectors are orthogonal (i.e., they have no similarity). 
# In the context of identity classification, cosine similarity can be used to compare feature vectors extracted from images to determine how closely they match.

def cosine_similarity(vec1, vec2):
    # epsilon (1e-6) to prevent division by zero
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (max(norm_vec1 * norm_vec2, 1e-6))
                              
# def cosine_similarity(a,b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    return img

