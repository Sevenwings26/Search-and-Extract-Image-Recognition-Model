# Search and Extract Image Recognition Model

A high-performance computer vision tool built to identify, match, and isolate specific individuals from complex group environments (image).

## Core Functions
* **Multi-Identity Detection:** Locates all human faces within a scene using the `RetinaFace` algorithm.
* **M:N Identity Resolution:** Support for both single-target search (1:N) and batch-identity extraction (M:N).
* **Similarity Scoring:** Provides a confidence score for each match, allowing for a user-defined threshold to prevent "False Positives."
* **Deep Feature Embedding:** Maps facial characteristics into a 512-dimensional vector space for high-precision matching.
* **Intelligent Extraction:** Automatically calculates bounding boxes with configurable padding (`MARGIN`) to produce professional-grade crops.
* **Hardware Agnostic:** Optimized for **CPU** via ONNX Runtime, with easy switching to **NVIDIA GPU** for production scaling.

## Libraries & Roles
* **`insightface`**: The core engine providing the `buffalo_l` model for detection and recognition.
* **`opencv-python`**: Handles image I/O (loading/saving) and the physical "cropping" of the image matrix.
* **`numpy`**: Handles the high-speed linear algebra (Dot Products and Norms) for identity similarity.
* **`onnxruntime`**: The cross-platform inference engine that executes the model's logic on your CPU(GPU replaceable).

---

## 📂 Project Structure

| File | Mode | Description |
| :--- | :--- | :--- |
| **`main1.py`** | **1 → N** | Takes a single-person query and finds the "Best Match" in a group photo. |
| **`main2.py`** | **N → N** | Processes multiple people in a query image and extracts each one found in the group photo into separate files. |
| **`utility.py`** | **Support** | Contains shared logic for image loading, normalization, and cosine similarity. |

---

## 🔧 Installation & Setup

1. **Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # venv\Scripts\activate on Windows
   ```

2. **Dependencies:**
   ```bash
   pip install opencv-python numpy insightface onnxruntime
   or
   pip install -r requirements.txt
   ```

3. **Execution:**
   Place your images in the project root and run:
   ```bash
   python main2.py
   ```
   *Extracted faces will be saved in the `/outputs` directory.*

---

## 🧠 Technical Highlights
* **Cosine Similarity via Dot Product:** By pre-normalizing vectors in `main2.py`, we calculate identity matches using simple dot products, significantly reducing CPU overhead during batch processing.
* **Spatial Clipping:** All extraction logic includes boundary-checking to ensure that crops do not exceed the original image dimensions, preventing "Index Out of Bounds" errors.
* **Margin Expansion:** Bounding boxes are dynamically expanded by a configurable `MARGIN` (default 20%) to include more context (hair/forehead) in the final output.

<!-- ---

### Suggested Next Addition:
Since you are a Backend/ML Engineer, you might eventually want to add a **"Performance"** section here. You could list the inference time on your local CPU (e.g., *"Processes a 10-person group in ~1.5s on Intel i7"*).  -->
