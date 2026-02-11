# Authentication for Wearable Devices

A machine learning system for authenticating users based on physiological signals captured from wearable devices. The system processes accelerometer and gyroscope data to extract biometric patterns, trains user-specific Convolutional Neural Networks (CNNs), and uses genetic algorithms to optimize model hyperparameters.

## Features

- **Biometric Authentication**: Uses physiological signals (BVP/BCG) from wearable sensors
- **CNN-Based Classification**: Deep learning models for binary classification (authentic user vs. imposters)
- **Genetic Algorithm Optimization**: Evolutionary hyperparameter tuning for optimal model performance
- **Modular Architecture**: Configurable CNN architectures and preprocessing pipelines
- **SQLite Data Storage**: Efficient storage and retrieval of processed sensor data
- **Comprehensive Evaluation**: TAR, TRR, FAR, FRR, and ROC curve metrics

## Architecture

### Data Flow
1. **Raw Data Collection**: Accelerometer and gyroscope readings from wearable devices (CSV format)
2. **Signal Processing**: Filtering and transformation into BVP (Blood Volume Pulse) or BCG (Ballistocardiography) signals
3. **Data Segmentation**: Fixed-time windows (configurable duration) for training samples
4. **Model Training**: User-specific CNN training with class-weighted binary classification
5. **Hyperparameter Optimization**: Genetic algorithm evolution of model architectures
6. **Authentication Testing**: Threshold-based decision making on sequential predictions

### Key Components

**Entry Point:**
- **`main.py`**: Command line interface for training, testing, and optimization

**Core Modules (`src/`):**
- **`params.py`**: Configuration for model parameters, data settings, and genetic algorithm (`GeneticConfig`)
- **`train.py`**: Trainer class for data preparation and model fitting
- **`test.py`**: Model evaluation with TAR/TRR metrics
- **`model.py`**: CNN model building, loading, and saving utilities
- **`genetic.py`**: Genetic algorithm implementation for hyperparameter optimization
- **`genome.py`**: Genome class representing model configurations for GA
- **`data_extraction.py`**: Data loading and tensor formatting from SQLite databases
- **`signal_filter.py`**: Signal processing filters for BVP/BCG extraction
- **`db.py`**: SQLite database interface for processed data storage
- **`loader.py`**: Database initialization from raw CSV files
- **`test_data.py`**: Test data preparation utilities
- **`visualization.py`**: Plotting utilities (ROC curves, training curves, GA analysis)

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow/Keras 2.x

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Authentication-For-Wearable-Devices
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare data:
   - Place raw sensor data in `data/{user_id}/{session_id}/{sequence_id}/` structure
   - Each sequence folder should contain `accel.csv` and `gyro.csv` files

4. Initialize databases:
```bash
# Process raw sensor data into SQLite databases (both BCG and BVP, all window sizes)
python main.py init-db
```
   - This creates databases for window sizes 2-5 seconds
   - Databases are stored in `databases_BCG/` and `databases_BVP/`

## Usage

### Command Line Interface

The CLI provides all functionality through subcommands:

```bash
# Initialize databases from raw sensor data
python main.py init-db                           # All window sizes (2-5), both BCG and BVP
python main.py init-db --signal-type bcg         # BCG only, all window sizes
python main.py init-db --signal-type bvp -w 4 5  # BVP only, specific sizes

# Train models for specific users
python main.py train --targets 9 10 11 12 --window-size 4 --epochs 100
python main.py train -t 9 10 11 -w 4 -e 50       # Short form
python main.py train -t 9 10 --signal-type bvp   # Train with BVP data

# Test trained models
python main.py test --targets 9 10 --signal-type bvp
python main.py test -t 9 10 11 -w 4 --threshold 0.5

# Run genetic algorithm optimization (all settings from GeneticConfig)
python main.py genetic

# Plot training curves from logs
python main.py plot-training --log-dir ./log_training --targets 2 3 4 5
python main.py plot-training -d ./log_training -t 2 3 4 -o curves.png

# Parse genetic algorithm logs
python main.py parse-logs --genetic-dir ./genetic_bcg_5
python main.py parse-logs -d ./genetic_bcg_5 -o boxplot.png
```


### Configuration

All configuration is centralized in `src/params.py`:

**GeneticConfig** - Genetic algorithm settings:
- `TRAITS_DICT`: Possible hyperparameter values (neurons, activations, dropout, etc.)
- `POPULATION_SIZE`: Number of genomes per generation (default: 10)
- `FITTEST_RATIO`: Percentage of top performers kept as parents (default: 0.25)
- `OGRES`: Random "losers" kept for genetic diversity (default: 1)
- `DEFAULT_MUTATION_RATE`: Mutation probability (default: 15%)
- `TARGETS_TEST`: User IDs for fitness evaluation
- `TRAIN_EPOCHS`: Training epochs per genome evaluation (default: 10)
- `TRAIN_SEGMENTS`: Training data segments (session_id, sequence_id) tuples
- `GENERATIONS`: Number of generations to evolve (default: 2)
- `WINDOW_SIZE`: Window size in seconds (default: 5)
- `SIGNAL_TYPE`: Signal type "bcg" or "bvp" (default: "bcg")

**Parameters** - Model and data settings:
- `window_sz`: Window size in seconds
- `hz`: Sampling frequency (default: 50 Hz)
- `signal_type`: BVP or BCG signal processing
- `model`: CNN architecture configuration dict
- `epochs`: Training epochs

## Data Format

### Raw Data Structure
```
data/
├── {user_id}/
│   ├── {session_id}/
│   │   ├── meta.txt
│   │   └── {sequence_id}/
│   │       ├── accel.csv
│   │       └── gyro.csv
```

### CSV Format
- `accel.csv`: Time, Ax, Ay, Az
- `gyro.csv`: Time, Gx, Gy, Gz
- Sampling frequency: 50 Hz (configurable)

### Processed Data
- Stored in SQLite databases as segmented tensors
- Shape: `(2, 3, window_sz * hz)` (2 signal types × 3 axes × time samples)

## Model Architecture

The CNN architecture is configurable through `params.model`:

```python
model_config = {
    # Convolutional layers (1 required, 2-5 optional)
    'conv1_neurons': 8,
    'conv1_activation': 'tanh',
    'conv2_enabled': True,
    'conv2_neurons': 256,
    'conv2_activation': 'tanh',
    # ...

    # Dense layers (all optional)
    'dense1_enabled': True,
    'dense1_neurons': 512,
    'dense1_activation': 'tanh',
    'dense1_dropout': 0.2,
    # ...

    # Output configuration
    'final_activation': 'tanh',
    'loss': 'mse',
    'optimizer': 'sgd'
}
```

## Genetic Algorithm

The genetic algorithm optimizes model hyperparameters by:

1. **Initialization**: Create population of random genomes (model configurations)
2. **Evaluation**: Train and test each genome across all target users
3. **Selection**: Keep top 25% as parents + 3 random "ogres" for diversity
4. **Breeding**: Parents produce offspring with inherited/mutated traits
5. **Iteration**: Repeat for N generations

Fitness score: `(FAR² + FRR²)` - lower is better

```bash
# Run genetic algorithm (all settings configured in GeneticConfig)
python main.py genetic
```

## Evaluation Metrics

- **TAR (True Accept Rate)**: Percentage of authentic users correctly accepted
- **TRR (True Reject Rate)**: Percentage of imposters correctly rejected
- **FAR (False Accept Rate)**: Percentage of imposters incorrectly accepted (100 - TRR)
- **FRR (False Reject Rate)**: Percentage of authentic users incorrectly rejected (100 - TAR)
- **AUC**: Area under the ROC curve
- **EER**: Equal Error Rate (where FAR = FRR)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

**[Full Paper](https://digital.wpi.edu/concern/student_works/02870z59d?locale=en)**

### MLA
> Bhatia, Meghana, et al. *Designing an Authentication System for Augmented Reality Devices*. Worcester Polytechnic Institute, 2019.

### BibTeX
```bibtex
@techreport{BhatiaGayosoKalampalikisAbualhaija2019,
    author = {Bhatia, Meghana and Gayoso, Alejandro Soler and Kalampalikis, Nikolaos and Abualhaija, Rushdi Mazen},
    title = {Designing an Authentication System for Augmented Reality Devices},
    institution = {Worcester Polytechnic Institute},
    year = {2019},
    url = {https://digital.wpi.edu/concern/student_works/02870z59d}
}
```
