# Real Estate Visual Learning Pipeline

---

# 1️⃣ Download Images

**Script**
`backend/scripts/download_kaggle_images.py`

**Command**

```bash
python download_kaggle_images.py
```

**Reads**

```
data/datasets/kaggle_prefiltered.csv
```

**Creates**

```
data/images/kaggle_raw/
```

**Purpose**
Downloads images incrementally without duplicates.

---

# 2️⃣ Visual Filtering (Scoring)

**Script**
`backend/scripts/filter_interiors.py`

**Command**

```bash
python filter_interiors.py
```

**Reads**

```
data/images/kaggle_raw/
```

**Creates / Updates**

```
data/datasets/interior_filter_pro.csv
```

**Purpose**
Analyzes images and generates an `indoor_score` based on visual features.

---

# 3️⃣ Semantic Filtering (YOLO)

**Script**
`backend/scripts/yolo_semantic_filter.py`

**Command**

```bash
python yolo_semantic_filter.py
```

**Reads**

```
data/datasets/interior_filter_pro.csv
```

**Creates**

```
data/datasets/interior_semantic.csv
```

**Purpose**
Adds semantic validation using YOLO object detection.

**Note**
YOLO is used only as a semantic signal layer, not as a final classifier.

---

# 4️⃣ Create Final Dataset (Visual + YOLO Fusion)

**Script**
`backend/scripts/create_final_dataset.py`

**Command**

```bash
python create_final_dataset.py
```

**Reads**

```
data/datasets/interior_semantic.csv
```

**Creates**

```
data/datasets/interior_final_candidates.csv
```

**Purpose**

Creates the final dataset used for learning.

Adds:

* `quality_bucket`
* `quality_bucket_human`
* `final_quality`

`final_quality` becomes the single source of truth for training.

Human labeling happens here.

---

# ==============================

# VISUAL LEARNING PIPELINE

# ==============================

# 5️⃣ Extract Visual Embeddings (OpenCLIP)

**Script**
`backend/scripts/extract_embeddings.py`

**Command**

```bash
python extract_embeddings.py
```

**Reads**

```
data/datasets/interior_final_candidates.csv
data/images/kaggle_raw/
```

**Creates**

```
data/embeddings/realestate_embeddings.parquet
```

**Purpose**

Encodes each image into a visual embedding using OpenCLIP.

This transforms the project from dataset engineering into representation learning.

---

# 6️⃣ Visualize Embedding Space (Diagnostic)

**Script**
`backend/scripts/visualize_embeddings.py`

**Command**

```bash
python visualize_embeddings.py
```

**Reads**

```
data/embeddings/realestate_embeddings.parquet
```

**Purpose**

Projects embeddings into 2D using UMAP to verify visual structure.

Diagnostic only. No files created.

---

# 7️⃣ Train Quality Head (Initial Learning Layer)

**Script**
`backend/scripts/train_quality_head.py`

**Command**

```bash
python train_quality_head.py
```

**Reads**

```
data/embeddings/realestate_embeddings.parquet
```

**Creates**

```
models/quality_head.joblib
```

**Purpose**

Learns initial mapping:

```
embedding → good / medium / bad
```

Used as pseudo-human baseline.

---

# 8️⃣ Human-in-the-loop Review (Ranking-Based)

**Script**
`backend/scripts/pseudo_human_loop.py`

**Command**

```bash
python pseudo_human_loop.py
```

**Reads**

```
realestate_embeddings.parquet
quality_head.joblib
```

**Purpose**

Shows most uncertain images first for efficient human correction.

Does NOT retrain automatically.

---

# ==============================

# RANKING ENGINE (CORE SYSTEM)

# ==============================

# 9️⃣ Train Pairwise Ranker

**Script**
`backend/scripts/train_pairwise_ranker.py`

**Command**

```bash
python train_pairwise_ranker.py
```

**Reads**

```
realestate_embeddings.parquet
```

**Creates**

```
models/pairwise_ranker.joblib
```

**Purpose**

Learns visual preference direction:

```
Image A > Image B
```

Transforms classification into ranking learning.

---

# 🔟 Fast Global Ranking (Production Method)

**Script**
`backend/scripts/build_fast_ranking.py`

**Command**

```bash
python build_fast_ranking.py
```

**Reads**

```
realestate_embeddings.parquet
pairwise_ranker.joblib
```

**Creates**

```
data/embeddings/realestate_fast_ranking.parquet
```

**Purpose**

Computes ranking score using:

```
fast_rank_score = embedding · w
```

This is the scalable ranking system used for production.

---

# ==============================

# ⚠️ IMPORTANT NOTES

# ==============================

* Images are never modified or moved by the pipeline.
* The system is CSV-driven — folder structure is not used for training logic.
* YOLO provides semantic signals only.
* `final_quality` is the only label used for learning.
* Embeddings are reused across all models.
* Pairwise ranking defines the global visual quality axis.
* Fast ranking replaces expensive pairwise comparisons.

---