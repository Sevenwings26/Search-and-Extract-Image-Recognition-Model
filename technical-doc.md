# Technical Documentation: Face-Search-Extract System

## 1. System Overview
The **Face-Search-Extract** system is a computer vision pipeline designed to solve the "needle in a haystack" problem for digital imagery. It identifies specific individuals in a crowded scene by comparing high-dimensional facial embeddings. 

The system is architected to run on **Commodity CPU Hardware**, utilizing the **ONNX Runtime** for optimized inference without the need for dedicated NVIDIA GPUs.

---

## 2. Architectural Pipeline
The system follows a modular, five-stage pipeline:

### Stage 1: Pre-processing & Detection
* **Engine:** `InsightFace` (utilizing the `RetinaFace` detectors).
* **Action:** The model scans the image for facial landmarks. It returns a `Face` object for every detected person, containing a Bounding Box (`bbox`), 5-point Landmarks (`kps`), and a probability score.

### Stage 2: Feature Extraction (Embedding)
* **Model:** `buffalo_l` (ResNet-based architecture).
* **Transformation:** Each detected face is aligned using the 5-point landmarks and passed through a Deep Convolutional Neural Network (DCNN).
* **Output:** A **512-dimensional feature vector** (Embedding). This vector represents the "mathematical fingerprint" of the face.

### Stage 3: Multi-Identity Resolution (Hungarian Algorithm)
When processing multiple queries against a group (**M:N**), the system avoids "Greedy" (first-come, first-served) matching to prevent **Identity Theft**.
* **Cost Matrix:** The system builds a matrix where each cell represents the "distance" ($1 - Similarity$) between a query and a group face.

$$Similarity = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

* **Optimization:** We implement the **Hungarian Algorithm** (via `scipy.optimize.linear_sum_assignment`).
* **Action:** The algorithm finds the global minimum cost, ensuring that the total similarity score for the entire group is mathematically maximized.
* **Thresholding:** A static constant `SIMILARITY_THRESHOLD = 0.5` is applied. Values above this are considered a "Match," while values below are discarded as noise or different identities.

### Stage 4: Spatial Extraction & Normalization
* **Margin Expansion:** Bounding boxes are dynamically expanded by a configurable `MARGIN` (default 20%) to provide a natural "portrait" crop.
* **Boundary Clipping:** Coordinates are clipped using `np.clip` to ensure they remain within the $(0, 0)$ and $(W, H)$ image boundaries.
* **Slicing:** Using NumPy, the group image is sliced and extracted.

### Stage 5: Persistence
* **Encoding:** The extracted matrices are converted to `.jpg` format.
* **Organization:** Files are saved to a dedicated `/outputs` directory with unique identifiers (e.g., `person_0.jpg`).

---

## 3. Technical Constraints & Decisions

### Why NumPy over PyTorch?
NumPy was chosen for the comparison layer because:
1.  **Low Latency:** Zero overhead compared to initializing heavy deep learning sessions.
2.  **Lightweight:** The total environment footprint is reduced by ~90% compared to a PyTorch installation.

### CPU vs. GPU Optimization
By using `ctx_id=-1` and `onnxruntime`, the system utilizes **AVX2/FMA instructions**. While slower than a GPU, it ensures the tool is highly portable across standard office hardware and cloud-based web servers.

---

## 4. Error Handling & Edge Cases

| Scenario | System Behavior | Mitigation Logic |
| :--- | :--- | :--- |
| **No Face in Query** | `ValueError` | Explicit check: `if not query_faces: raise` |
| **Out-of-Bounds Bbox** | Matrix slicing error | `np.clip` keeps coordinates inside image dimensions. |
| **Identical Lookalikes** | Ambiguous Matching | Resolved by Hungarian Global Optimization. |
| **Low Resolution** | High False Negatives | Minimum detection size is adjustable via `det_size`. |

---

## 5. Future Scalability (Roadmap)

As the search space expands from a single image to a massive database (10,000+ faces), the architecture will evolve into a **Tiered Retrieval System**:

### Tier 1: Retrieval (FAISS)
* **Mechanism:** Facebook AI Similarity Search (FAISS) using **Inverted File Indexes**.
* **Role:** Acts as a "High-Speed Filter" to find the **Top-K** candidates in milliseconds.


### Tier 2: Refinement (Hungarian)
* **Role:** Acts as the "Precision Judge."
* **Action:** The Hungarian Algorithm is applied *only* to the small subset returned by FAISS, ensuring the final assignment is 100% unique and globally optimal.

---

### **Summary of Evolutionary Path**

| Feature | Local Inference (Current) | Enterprise Scaling (Future) |
| :--- | :--- | :--- |
| **Search Scope** | Single Group Photo | Massive Image/Video Database |
| **Search Engine** | NumPy Linear Scan | **FAISS Vector Indexing** |
| **Complexity** | $O(N)$ | **$O(\log N)$** |
| **Assignment** | Greedy or Basic Hungarian | **Hybrid FAISS + Hungarian** |

---

**Developer Note:** *This system is currently in its M:N optimization phase. Transitioning to Tier 1 Scaling should be triggered once the local similarity matrix computation exceeds a 500ms latency threshold.*
