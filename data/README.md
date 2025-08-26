# Data Directory Structure

## Organization
- `raw/`: Unprocessed data as received from sources
- `processed/`: Cleaned and preprocessed data ready for training
- `train/`: Training datasets
- `test/`: Test datasets for final evaluation
- `validation/`: Validation datasets for hyperparameter tuning

## Data Types
Each subdirectory contains folders for different data modalities:
- `images/`: Image datasets for computer vision tasks
- `text/`: Text datasets for NLP tasks (classification, generation)
- `tabular/`: Structured data for traditional ML tasks

## Guidelines
1. Keep raw data immutable - never modify original files
2. Document preprocessing steps in accompanying .md files
3. Use consistent naming conventions: dataset_version_split.format
4. Include data manifests (file lists and metadata) for reproducibility