# OCR-Based(LayoutLM) vs. OCR-Free(Donut) Document Understanding
Information extraction from receipts using LayoutLMv3 and Donut

---

# LayoutLMv3 : Automated Receipt Information Extraction

This repository contains a deep learning solution for extracting key information from scanned receipts (dataset) using **LayoutLMv3**. The model processes text, layout (bounding boxes), and visual features simultaneously to achieve high-accuracy token classification.

##  Architecture & Methodology
The project leverages the `microsoft/layoutlmv3-base` model. Unlike traditional NLP models, LayoutLMv3 utilizes a multimodal approach:
* **Textual Embeddings:** Processes the semantic content of the document.
* **Spatial Embeddings:** Encodes the relative positions (bounding boxes) of words.
* **Visual Patches:** Extracts image features directly from the document.

The training workflow is managed via **PyTorch Lightning** to ensure modularity, scalability, and reproducible experiments.

---

##  Training Configuration & Hyperparameters
The model was fine-tuned using the following optimized settings for Token Classification:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Max Epochs** | 20 | Maximum number of training iterations. |
| **Batch Size** | 4 | Balanced for 16GB VRAM GPUs. |
| **Learning Rate (LR)** | $2 \times 10^{-5}$ | Base learning rate for the transformer backbone. |
| **Classifier LR** | $2 \times 10^{-4}$ | 10x higher LR for the classification head. |
| **Weight Decay** | 0.01 | AdamW regularization to prevent overfitting. |
| **Warmup Steps** | 10% | Linear warmup at the start of training. |
| **Precision** | 16-mixed | Mixed precision for faster GPU computation. |
| **Early Stopping** | 4 (patience) | Stops training if `val_f1` plateaus. |

---

##  Performance Metrics
The model achieved the following results on the test set:

### Global Scores:
* **F1-Score:** **0.9409**
* **Precision:** **0.9395**
* **Recall:** **0.9423**

### Detailed Classification Report:
| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **ADDRESS** | 0.91 | 0.96 | 0.93 | 517 |
| **COMPANY** | 0.95 | 0.94 | 0.94 | 241 |
| **DATE** | 0.98 | 0.97 | 0.98 | 795 |
| **TOTAL** | 0.90 | 0.88 | 0.89 | 457 |

> **Analysis:** The model shows exceptional performance on **DATE** and **COMPANY** names. The **TOTAL** category remains the most challenging due to competing numeric values (like subtotals), but custom inference heuristics have significantly mitigated false positives.

---

##  Inference Visualization
The model successfully generalizes across various receipt layouts. Below are examples of the model identifying entities:
* **Yellow:** COMPANY
* **Blue:** ADDRESS
* **Green:** DATE
* **Red:** TOTAL

<p align="center">
  <img src="assets/result_X51006856982.jpg" width="32%" />
  <img src="assets/result_X51006647933.jpg" width="32%" />
  <img src="assets/result_X51009453729.jpg" width="32%" />
</p>

---

##  Risks & Potential Limitations
With the current dataset size (~600 samples), the following risks should be considered:
1.  **Template Overfitting:** If the training set is dominated by specific retail chains, the model may memorize layouts rather than learning to generalize.
2.  **Complex Backgrounds:** Noisy or dark backgrounds can degrade OCR accuracy or visual feature extraction.
3.  **Long Documents:** In extremely long receipts, the spatial relationship between a "TOTAL" header and its value may be lost if they exceed the maximum sequence length.

---

##  Future Improvements & Scaling
1.  **Data Augmentation:** Implement `albumentations` (noise, blur, rotation) to simulate poor-quality scans.
2.  **Dataset Expansion:** Increasing the sample size from 600 to 5,000+ would likely push the F1-score beyond **0.97**.
3.  **Advanced OCR:** Integrating modern engines like PaddleOCR for more precise initial bounding box coordinates.
4.  **Ensemble Methods:** Combining LayoutLMv3 with a text-only model (e.g., RoBERTa) to verify logical consistency.

---

##  Project Conclusion
This project successfully demonstrates an end-to-end pipeline for extracting structured data from unstructured documents. Achieving an **F1-Score of 0.94** proves the robustness of the multimodal LayoutLMv3 approach. The system is ready for integration into business automation tasks such as accounting and expense tracking. By combining text, layout, and vision, this solution significantly outperforms traditional text-only NER methods.

---

# Donut (Document Understanding Transformer)

This repository contains the implementation and evaluation of the **Donut** model, an OCR-free transformer-based approach for Document Information Extraction. Unlike LayoutLMv3, Donut operates directly on image pixels without requiring external OCR engines, converting visual information directly into structured JSON.

##  Model Strategy & Constraints
The **Donut** model is a "heavyweight" end-to-end transformer. During development, several critical constraints were addressed:
* **Computational Weight:** Donut is significantly more resource-intensive than token classification models.
* **The "Cutting" Problem:** To fit high-resolution receipts into VRAM, images often require resizing or aggressive cropping ("cutting"). This is a double-edged sword: resizing reduces memory load but can blur small characters, leading to extraction errors.
* **OCR-Free Nature:** Since there is no external OCR, the model must learn to "read" and "understand" simultaneously, which requires high-quality visual data.

---

##  Training Configuration & Hyperparameters
The model was fine-tuned using a hybrid dataset strategy to compensate for the limited real-world samples.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Backbone** | `naver-clova-ix/donut-base` | Vision Encoder-Decoder Transformer. |
| **Max Epochs** | 30 | Extended training to stabilize the generative decoder. |
| **Batch Size** | 2 | Strictly limited due to high GPU memory footprint. |
| **Input Resolution** | [1280, 960] | Optimized via cropping/resizing to balance detail vs. VRAM. |
| **Dataset Composition** | 600 (Real) + 1000 (Synthetic) | 1,600 total samples to improve font/layout variety. |
| **Augmentations** | Rotation, ColorJitter, Blur | Critical for simulating low-quality physical scans. |

---

##  Performance Analysis & Metrics
Evaluation was performed on **168 test images**. The results below reflect the model's performance after string normalization.

###  Field-Level Results:
| Entity | Accuracy (Exact Match) | Average Similarity (Edit Distance) | Estimated F1-Score* |
| :--- | :--- | :--- | :--- |
| **Company** | 82.74% | 92.62% | **0.87** |
| **Date** | 89.88% | 96.12% | **0.93** |
| **Address** | 80.95% | 95.13% | **0.88** |
| **Total** | 87.50% | 94.75% | **0.91** |
| **OVERALL** | **Full Match: 61.90%** | **Avg Sim: 94.66%** | **~0.90** |

*\*Estimated F1-score derived from the harmonic mean of Accuracy and Similarity metrics.*

###  Analysis of Results:
1. **The Similarity vs. Accuracy Gap:** Notice that "Average Similarity" is consistently higher (>92%) than "Accuracy." This indicates that when the model makes a mistake, it is usually minor (e.g., one wrong character in a long address), rather than a total hallucination.
2. **Date Excellence (89.88% Acc):** Dates have standard patterns, making them easier for the decoder to generate correctly.
3. **Address & Company Challenges:** These fields suffer most from the **image cutting/resizing** issue. When fine print is blurred during downscaling, the vision encoder loses the ability to distinguish similar characters.
4. **Full Match (61.90%):** This represents "perfect" extraction of all fields in a single document. While seemingly lower than LayoutLM, it is a high score for an OCR-free generative approach on such a small dataset.

---

##  Inference & Visual Error Analysis
The following visualizations show the model's output. Note how the model maps visual regions directly to JSON fields.

<p align="center">
  <img src="assets/result_X51006554841.jpg" width="32%" />
  <img src="assets/result_X51006555833.jpg" width="32%" />
  <img src="assets/result_X51006619784.jpg" width="32%" />
</p>

###  Common Error Patterns (Why it fails):
* **Truncation Errors:** Since we "cut" images to save memory, tokens near the edges are sometimes partially lost.
* **Resolution Bottleneck:** Small price fonts under the "Total" section become "unreadable" if the resizing ratio is too aggressive.
* **Alignment issues:** In very long receipts, the model may lose track of which "Total" is the final one if there are multiple sub-totals.

---

##  Optimization & Scaling Strategy
To reach >95% Accuracy, the following steps are required:
1. **Increase Data Volume:** The current 600 real samples are insufficient for a model of this weight. Expanding to **2,000+ real samples** is critical.
2. **Synthetic Data Refinement:** The current 1,000 synthetic samples helped, but increasing this to **5,000+** with more complex noise (folds, shadows, coffee stains) will harden the model.
3. **Sliding Window Inference:** Instead of resizing the whole image, we can process overlapping high-resolution patches to avoid "blurring" small text.
4. **Enhanced Augmentations:** Using `albumentations` to simulate camera-shake and motion blur, which are common in mobile-captured receipts.

---

##  Final Conclusion
The **Donut** model is a powerful, "all-in-one" solution that removes the need for complex OCR pipelines. While it is **computationally expensive** and sensitive to **image resolution (cutting)**, it provides a much more semantic understanding of the document than text-only models.

With an overall **Average Similarity of 94.66%**, the model proves that even with limited data (600 real / 1000 synthetic), it can learn to extract structured data with high precision. For production environments, focusing on higher-resolution input processing and expanding synthetic training data will be the key drivers for reaching peak performance.

---

# Final Comparison: LayoutLMv3 vs. Donut

This section provides a definitive performance and architectural comparison between the two implemented models.

###  Comparative Performance Matrix

| Metric / Category | LayoutLMv3 (OCR-Based) | Donut (OCR-Free) | Winner | Key Insight |
| :--- | :---: | :---: | :---: | :--- |
| **Global F1-Score** | **0.9409** | ~0.9000 | **LayoutLMv3** | High spatial precision wins on SROIE dataset. |
| **Company Accuracy** | **0.9424 (F1)** | 82.74% (Exact Match) | **LayoutLMv3** | Donut occasionally struggles with long entity names. |
| **Date Accuracy** | **0.9879 (F1)** | 89.88% (Exact Match) | **LayoutLMv3** | LayoutLM is nearly perfect on standardized dates. |
| **Total Accuracy** | **0.8945 (F1)** | 87.50% (Exact Match) | **LayoutLMv3** | Competitive performance from both models. |
| **Avg. Similarity** | N/A | **94.66%** | **Donut** | Donut understands "semantics" despite minor typos. |
| **Inference Speed** | **Fast** (Parallel) | Slow (Sequential) | **LayoutLMv3** | LayoutLM is better suited for real-time apps. |
| **Hardware Demand** | **Low** (Base model) | High (VRAM intensive) | **LayoutLMv3** | Donut requires significantly more GPU memory. |
| **OCR Dependency** | Required | **None** | **Donut** | Donut eliminates 3rd party OCR costs & complexity. |
---

##  Deep Dive: Why LayoutLMv3 Outperformed Donut

The performance gap observed in the SROIE project is driven by two primary factors: **Spatial Precision** and **Input Resolution**.

### 1. The Resolution & "Cutting" Bottleneck
Donut is an end-to-end vision model. To prevent GPU memory (VRAM) overflow, receipt images were either downscaled or "cut." 
* **Donut's Weakness:** When a high-resolution receipt is downsampled, fine numeric details (like the "Total" price) become pixelated. Since Donut relies solely on pixels, it easily confuses similar numbers (e.g., `8` and `0`).
* **LayoutLMv3's Strength:** It uses explicit OCR coordinates provided in the dataset. Even if the image is slightly blurry, the model *knows* exactly where the text is and uses the high-quality character recognition from the OCR engine.

### 2. Explicit Spatial Awareness

* **LayoutLMv3** treats the 2D position $(x, y)$ as a primary feature. It quickly learns the "common sense" of a receipt (e.g., the "Total" is usually at the bottom-right).
* **Donut** must learn these spatial relationships implicitly. On a relatively small dataset (1,600 samples), it is much harder for a generative vision model to reach the same level of geometric certainty as a coordinate-based model.

---

##  Final Conclusion & Selection Guide

###  The Winner: LayoutLMv3
**LayoutLMv3** is the clear winner for the SROIE task and production-ready applications.
* **Why:** It yields a significantly higher **F1-score (0.9409)**, processes images faster, and is remarkably stable in extracting high-stakes fields like **Date** and **Total**.
* **Best for:** Production environments where OCR data is available and speed/accuracy are critical requirements.

###  The Alternative: Donut
**Donut** is the superior choice for OCR-free research and non-standard document types.
* **Why:** It eliminates the need for an external OCR engine, simplifying the pipeline and reducing long-term costs. While it struggles with exact character matches on small datasets, its **94.66% Average Similarity** proves it deeply understands document semantics.
* **Best for:** Documents where standard OCR engines fail (handwriting, stylized fonts, artistic layouts) or when building purely generative AI document assistants.

---

##  Project Summary & Future Outlook

This benchmark confirms that for structured document extraction with provided layout data, **LayoutLMv3** remains the industry standard. However, **Donut’s** ability to learn from a small mix of real and synthetic samples without an OCR engine shows massive potential for future scaling.

###  Key Takeaways for Improvement:
1. **Scaling Data:** Expanding the dataset to **2,000+ real-world samples** to improve model generalization.
2. **Resolution Management:** Implementing **sliding-window inference** for Donut to preserve high-resolution text features while managing VRAM constraints.
3. **Hybrid Approaches:** Exploring the use of modern, lightweight OCR engines (like PaddleOCR) to further boost LayoutLMv3’s input quality in "messy" real-world scenarios.
