# Fall Detection Using Computer Vision and Machine Learning

This repository contains the implementation of our course project for
*Machine Learning Principles*, focusing on fall detection using
computer vision and temporal modeling techniques.

## Project Overview
Falls are a major patient safety issue, especially among elderly and
post-operative patients. This project explores whether fall-related
patterns can be captured from visual data using deep learning and
machine learning models.

## Methods
- Human pose estimation using YOLO Pose
- CNN baseline for frame-level fall classification
- Temporal modeling:
  - Sliding Window Soft Voting
  - CNN + LSTM
  - CNN + GRU
- Machine learning models:
  - Random Forest
  - XGBoost
  - LightGBM

## Dataset
We use a public fall detection dataset from Kaggle.
Due to licensing and ethical considerations, raw video/image data are
not included in this repository.

See `data/README.md` for details.

## Repository Structure
- `models/` : All model implementations
- `results/` : Experimental figures and outputs
- `ethics/` : Research ethics statement

## Disclaimer
This repository is for academic and educational purposes only.
The models are not intended for clinical deployment.
