# Technical Documentation: Face-Search-Extract System

## 1. System Overview
The **Face-Search-Extract** system is a computer vision pipeline designed to solve the "needle in a haystack" problem for digital imagery. It identifies a specific individual in a crowded scene by comparing high-dimensional facial embeddings.

The system is architected to run on **Commodity CPU Hardware**, utilizing the **ONNX Runtime** for optimized inference without the need for dedicated NVIDIA GPUs.

---

## 2. Architectural Pipeline
The system follows a linear, four-stage pipeline:

### Stage 1: Pre-processing & Detection
* **Engine:** `InsightFace` (utilizing the `RetinaFace` or `SCRFD` detectors).
* **Input:** Raw RGB Image.
* **Action:** The model scans the image for facial landmarks. It returns a `Face` object for every detected person, containing a Bounding Box (`bbox`), 5-point Landmarks (`kps`), and a probability score.

### Stage 2: Feature Extraction (Embedding)
* **Model:** `buffalo_l` (ResNet-based architecture).
* **Transformation:** Each detected face is cropped, aligned, and passed through a Deep Convolutional Neural Network (DCNN).
* **Output:** A **512-dimensional feature vector** (Embedding). This vector represents the "mathematical fingerprint" of the face, where similar faces are located closer together in vector space.

### Stage 3: Vector Comparison (Similarity Logic)
To determine if the `Query Face` exists in the `Group Photo`, we calculate the **Cosine Similarity** between the query vector ($A$) and every group vector ($B$):

$$Similarity = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

* **Thresholding:** A static constant `SIMILARITY_THRESHOLD = 0.5` is applied. Values above this are considered a "Match," while values below are discarded as noise or different identities.

### Stage 4: Spatial Extraction
* **Coordinate Mapping:** Once the "Best Match" index is identified, the system retrieves the `bbox` coordinates.
* **Array Slicing:** Using NumPy, the group image (represented as a 3D Matrix) is sliced: `image[y1:y2, x1:x2]`.
* **Persistence:** The resulting sub-matrix is encoded into a `.jpg` or `.png` format via OpenCV.

---

## 3. Technical Constraints & Decisions

### Why NumPy over Deep Learning Frameworks?
During the inference phase, the bottleneck is the model execution (handled by ONNX), not the vector math. NumPy was chosen for:
1.  **Low Latency:** Zero overhead compared to initializing PyTorch/TensorFlow sessions.
2.  **Memory Footprint:** Keeps the application lightweight (~15MB vs ~2GB for PyTorch).
3.  **Portability:** Native support for the Python scientific stack.

### CPU Optimization
By using `ctx_id=-1` and `onnxruntime`, the system utilizes **AVX2/FMA instructions** on modern CPUs. While slower than a GPU ($100ms$ vs $1500ms$ per image), it ensures the tool can be deployed on standard web servers and laptops.

---

## 4. Error Handling & Edge Cases

| Scenario | System Behavior | Mitigation Logic |
| :--- | :--- | :--- |
| **No Face in Query** | `ValueError` | `if not query_faces: raise` |
| **Multiple Query Faces** | Index `[0]` selection | Documentation instructs users to provide single-person crops for queries. |
| **Out-of-Bounds Bbox** | Matrix slicing error | `np.clip` ensures coordinates stay within `(0, 0)` and `(W, H)`. |
| **Low Resolution** | High False Negatives | Minimum detection size can be adjusted in `app.prepare`. |

---

<!-- ## 5. Future Scalability
* **Face Alignment:** Using the 5-point landmarks to perform an affine transformation before extraction to ensure upright head positioning.
* **Batch Processing:** Modifying the pipeline to accept a directory of images and performing a "one-to-many" search across thousands of files.
* **Vector Database:** For production use (10,000+ faces), replacing the linear NumPy loop with a vector database like **FAISS** or **Milvus** for $O(log n)$ search speeds. -->
