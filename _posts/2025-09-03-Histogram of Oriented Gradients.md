---
layout: post
title: "Histogram of Oriented Gradients"
date: 2025-03-09 10:00:00 +0700
categories: [Computer Vision]
tags: [CV,AI]
math: true
# author: "Your Name"
---

# Histogram of Oriented Gradients (HOG)

The **Histogram of Oriented Gradients (HOG)** is a feature descriptor used in computer vision for object detection and recognition. It represents objects by computing the distribution of gradient orientations in localized image regions.

## How HOG Works (Step-by-step)

### 1. Gradient Calculation (Sobel Operator)
To detect edges, gradients are computed using the **Sobel operator**:

**Horizontal Gradient (Gₓ)**:
```py
[-1 0 1
 -2 0 2 
 -1 0 1]
```
**Vertical Gradient (Gᵧ)**:
```py
[-1 -2 -1
 0 0 0
  1 2 1]
```

For each pixel (x, y), compute:
- **Gradient Magnitude**:  
  $Magnitude = \sqrt{Gx² + Gy²}$
- **Gradient Direction**:  
  $Theta = \arctan(Gy / Gx)$

Gradient directions are typically **unsigned (0°-180°)** or **signed (0°-360°)**.

---

### 2. Creating Cells
- The image is divided into **cells** (typically **8×8 pixels**).
- Each cell will have a histogram of gradient directions.

---

### 3. Histogram Creation
- Each gradient contributes to a **9-bin histogram** (e.g., bins covering **0°-180°** in **20° intervals**).
- The gradient magnitude is used as a weight when voting into bins.

**Example:**
If a pixel has:
- Gradient magnitude = 8.5
- Gradient angle = 30°

Then, the value **8.5** contributes to the **[20°-40°] bin**.

---

### 4. Block Normalization
- Cells are grouped into **blocks** (e.g., **2×2 cells = 16×16 pixels**).
- Normalize the feature vectors within each block to improve robustness against changes in lighting.

**L2-norm normalization** formula:
$$
  v_{normalized} = \frac{v}{ \sqrt{(||v||² + ε²)}}
$$

Where:
- v = histogram vector
- ε = small constant (e.g., **0.0001**) for stability.

---

### 5. Feature Vector Construction
- The normalized histograms from all blocks are **concatenated** into a single **feature vector**.
- This vector is used for **object detection or classification**.

---

## Example Calculation

Consider a **16×16 pixel grayscale image region** (usually HOG is applied on larger images):

1. **Compute Gradients**: Using Sobel filters.
   - Example: At pixel (x, y), suppose:
     - Gradient magnitude = 6
     - Gradient direction = 45°

2. **Divide into Cells**: Each **8×8 pixels**.

3. **Compute 9-bin Histograms per Cell**:
   - Example bin ranges: **[0°-20°], [20°-40°], ..., [160°-180°]**
   - A pixel with a **45°** gradient contributes its **magnitude (6)** to the **[40°-60°] bin**.

4. **Normalize these histograms within overlapping blocks (2×2 cells)**.

5. **Concatenate all histograms into a final descriptor vector** → Used by a machine learning classifier (e.g., **SVM**).

---

## Key Parameters

| Parameter            | Typical Choice                             |
| -------------------- | ------------------------------------------ |
| Gradient Computation | Sobel operator (Prewitt, Scharr also used) |
| Cell Size            | **8×8 pixels**                             |
| Orientation Bins     | **9 bins** (0°-180°)                       |
| Block Size           | **2×2 cells (16×16 pixels)**               |
| Block Stride         | **8 pixels** (typically 50% overlap)       |
| Normalization        | **L2-norm or L2-Hys**                      |

---

## Advantages & Limitations

### ✅ Advantages:
- **Captures edge and shape information effectively**.
- **Robust to illumination changes**.
- **Performs well for object detection (e.g., pedestrians, faces)**.

### ❌ Limitations:
- **Computationally expensive** compared to simpler descriptors.
- **Sensitive to large scale variations and occlusions**.

---

## Applications
- **Pedestrian detection** 🏃
- **Human detection** 👤
- **Face recognition** 🤖
- **Object recognition** 📷

---

## Summary
HOG is a **powerful feature descriptor** for object detection. It captures **gradient orientations** in small image regions, normalizes them, and produces a feature vector used for machine learning.
