# Closed-Loop AI Enables Organic Continuous-Wave Laser

This repository contains the source code for the research article  
**"Closed-Loop AI Enables Organic Continuous-Wave Laser"**.

It includes:
- A **predictive model** for photoluminescence quantum yield (PLQY) and lasing performance.
- A **generative model** for molecular design.
- Example datasets and scripts to reproduce the inference process.

---

## Repository Structure

```plaintext
.
├── pred_plqy/              # PLQY prediction task
│   ├── infer.py            # Inference script for PLQY
│   ├── infer.sh            # Example usage with standard values in comments
│   └── valid.lmdb          # Small PLQY dataset
│
├── pred_laser/             # Laser performance prediction task
│   ├── infer.py            # Inference script for laser prediction
│   ├── infer.sh            # Example usage with standard values in comments
│   └── valid.lmdb          # Small laser dataset
│
├── unimol-gen/             # Molecular generative model
│
├── unimol-pre/             # Laser performance prediction model
│
├── weights_laser/          # Weight of laser
│
├── weights_plqy/           # Weight of plqy
│
└── README.md               # This documentation
```

## Environment Setup
This project relies on the Uni-Mol framework for molecular representation learning.
To set up the environment:

Clone the Uni-Mol repository:

```bash
git clone https://github.com/deepmodeling/Uni-Mol.git
```
Follow the official Uni-Mol setup guide to install dependencies and prepare the environment.

## Download weights
https://drive.google.com/drive/folders/1fac-PKyMWZBOKEi2JNqOLdUcvZCZRLJy?usp=sharing

## Run Inference
After setting up the environment, you can perform inference using the provided scripts.
### Predict PLQY
```bash
cd pred_plqy
python infer.py
```
### Predict Laser
```bash
cd pred_laser
python infer.py
```
Prediction results will be saved in the output files. Reference values are included in the comments of infer.sh

## Related Work
This work builds on our previous publication:

Uni-Mol: A Universal 3D Molecular Representation Learning Framework
https://openreview.net/forum?id=6K2RM6wVqKu

Ou, Q., Peng, Q. & Shuai, Z. Computational screen-out strategy for electrically pumped organic laser materials. 
Nat Commun 11, 4485 (2020).
https://doi.org/10.1038/s41467-020-18144-x
