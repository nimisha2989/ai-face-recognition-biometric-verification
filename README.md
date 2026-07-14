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
