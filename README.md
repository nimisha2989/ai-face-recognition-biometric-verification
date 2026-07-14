# AI Face Recognition & Biometric Verification System

## Project Overview

This project presents a deep learning-based face recognition and biometric verification system designed to evaluate both identity verification and identification tasks.

The system uses a fine-tuned ResNet18 convolutional neural network to extract 512-dimensional facial embeddings. Cosine similarity is used to compare facial representations and determine identity similarity.

The project also explores PCA-based dimensionality reduction and score-level fusion between ResNet18 and MobileNetV2 to evaluate system performance, computational efficiency, and generalization to unseen identities.


## Problem Statement

Face recognition systems must perform reliably across variations in pose, lighting conditions, facial expressions, and unseen identities. This project investigates deep learning-based approaches for building a robust biometric system capable of performing two core tasks:

- **Face Verification (1:1 Matching):** Determines whether two facial images belong to the same individual.
- **Face Identification (1:N Matching):** Identifies an individual by comparing a facial image against a gallery of known identities.

The project evaluates model performance in both subject-dependent and subject-independent scenarios to study generalization to unseen identities.

## Key Features

- Fine-tuned **ResNet18** using transfer learning for facial recognition.
- Extracted **512-dimensional deep facial embeddings**.
- Implemented **cosine similarity** for biometric matching.
- Evaluated **1:1 face verification** and **1:N face identification**.
- Applied **Principal Component Analysis (PCA)** to reduce embedding dimensions from 512 to 128.
- Implemented **score-level fusion using ResNet18 and MobileNetV2**.
- Evaluated performance using **ROC-AUC, Equal Error Rate (EER), TAR@FAR, and Rank-1 Accuracy**.
- Tested generalization using **unseen identities**.


## System Architecture

The biometric recognition pipeline follows the workflow below:

1. **Face Image Input**  
   Facial images are loaded from the Georgia Tech Face Database.

2. **Image Preprocessing**  
   Images are resized and normalized using ImageNet preprocessing parameters.

3. **Deep Feature Extraction**  
   A fine-tuned ResNet18 model extracts a 512-dimensional embedding representing facial characteristics.

4. **Embedding Comparison**  
   Cosine similarity is used to measure similarity between facial embeddings.

5. **Biometric Decision**
   - **Verification:** Determines whether two images belong to the same identity.
   - **Identification:** Compares a probe image against a gallery of enrolled identities.

6. **Performance Evaluation**  
   The system is evaluated using ROC-AUC, Equal Error Rate (EER), TAR@FAR, and Rank-based identification accuracy.

7. **Optimization and Model Fusion**  
   PCA is used for dimensionality reduction, and ResNet18 scores are fused with MobileNetV2 scores to evaluate multi-model biometric performance.


## Results and Key Findings

The system was evaluated across subject-dependent verification, face identification, subject-independent generalization, dimensionality reduction, and multi-model score fusion.

| Experiment | Result | Key Observation |
|---|---|---|
| Subject-Dependent Verification | ROC AUC: 0.9995 | Strong separation between genuine and impostor scores |
| Verification Score Separation | d-prime: 8.83 | Highly discriminative facial embeddings |
| Rank-1 Identification | 98% | Correct identity retrieved as the top match in most cases |
| Rank-3 Identification | 98% | Correct identity appeared within the top three matches |
| Subject-Independent Verification | ROC AUC: 0.8986 | Unseen identities presented a more challenging generalization scenario |
| PCA Dimensionality Reduction | 512 → 128 dimensions | Retained approximately 99.4% of the original variance |
| PCA Verification | ROC AUC: 0.9969 | Strong verification performance maintained after compression |
| Multi-Model Score Fusion | ROC AUC: 0.99888 | ResNet18 and MobileNetV2 score fusion provided robust verification performance |

### Key Findings

- Transfer learning with ResNet18 produced highly discriminative facial embeddings even with a limited face dataset.
- Subject-dependent verification achieved strong genuine and impostor score separation.
- Subject-independent testing demonstrated the increased difficulty of biometric recognition on unseen identities.
- PCA reduced embedding dimensionality by 75% while maintaining strong verification performance.
- Score-level fusion combined complementary similarity information from ResNet18 and MobileNetV2.

  ## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- MobileNetV2
- Scikit-learn
- NumPy
- Matplotlib
- Google Colab
- Jupyter Notebook

## Machine Learning and AI Techniques

- Deep Learning
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Deep Feature Embeddings
- Cosine Similarity
- Biometric Verification
- Face Identification
- Principal Component Analysis (PCA)
- Score-Level Fusion
- ROC and AUC Analysis
- Equal Error Rate (EER)
- Cumulative Match Characteristic (CMC) Analysis

  ## How to Run the Project

1. Download or clone this repository.
2. Open `face_recognition_biometric_verification.ipynb` in Google Colab or Jupyter Notebook.
3. Download the Georgia Tech Cropped Face Database.
4. Upload the dataset ZIP file when prompted by the notebook.
5. Run the notebook cells sequentially to:
   - Extract and inspect the dataset.
   - Prepare training and testing samples.
   - Fine-tune the ResNet18 model.
   - Extract deep facial embeddings.
   - Evaluate face verification and identification performance.
   - Perform subject-independent evaluation.
   - Apply PCA dimensionality reduction.
   - Train MobileNetV2 and evaluate score-level fusion.

## Project Author

**Nimisha Chandra**

Master's in Electrical and Computer Engineering  
AI/ML-focused technical and project professional
