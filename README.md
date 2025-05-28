# Breast Cancer Subtype Classification

This project implements deep learning (CNNs) and gradient-boosted tree models to identify breast cancer subtypes from medical imaging data.

## Project Overview

The project aims to:
- Classify breast cancer subtypes using medical imaging data
- Compare performance between CNNs and gradient-boosted tree models
- Provide interpretable results through explainable AI techniques

## Project Structure

```
.
├── data/                    # Data directory (not tracked in git)
│   ├── raw/                # Raw input data
│   └── processed/          # Processed data
├── notebooks/              # Jupyter notebooks
│   └── data_requirements.ipynb  # Documentation of required data
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/              # Data processing scripts
│   ├── models/            # Model implementations
│   └── utils/             # Utility functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Required Data

See `notebooks/data_requirements.ipynb` for detailed information about:
- Required data formats
- Data preprocessing steps
- Data sources and acquisition

## Models

The project implements two main approaches:
1. Convolutional Neural Networks (CNNs)
   - Architecture optimized for medical imaging
   - Transfer learning from pre-trained models
   
2. Gradient Boosted Trees
   - XGBoost/LightGBM implementation
   - Feature engineering pipeline

## Contributing

Please read the data requirements notebook before contributing to ensure all necessary data formats and preprocessing steps are followed.

## License

MIT License
