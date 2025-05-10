# PointNetLK_Revisited


This repository is dedicated to comparing different Feature Extractors.  
**Please focus on the `feature_extraction.py` file**, which provides the core implementation for all feature extractors.

## Overview

The `feature_extraction.py` file contains:
- A base Feature Extractor class
- All implemented Feature Extractors, including the original PointNet and its variants

Each Feature Extractor consists of three main components:
1. **Model Block Definition**  
   The architecture and layers of the feature extractor model.
2. **Feature Extractor Class Creation**  
   The class that encapsulates the model and its forward logic.
3. **Feature Jacobian Computation**  
   The method for calculating the Jacobian matrix of the features.

## Structure

- `feature_extraction.py`  
  - Base class for feature extractors  
  - Implementations of PointNet and all its variants  
  - Each extractor includes model definition, class instantiation, and Jacobian computation

## Usage

To use or extend a feature extractor, simply refer to the corresponding class in `feature_extraction.py`.  
You can add new variants by following the structure of the existing implementations.

## Purpose

The main goal of this repository is to provide a unified and extensible framework for benchmarking and comparing different feature extraction methods, especially in the context of PointNet and its derivatives.

---

Feel free to open issues or pull requests for improvements or new feature extractors.