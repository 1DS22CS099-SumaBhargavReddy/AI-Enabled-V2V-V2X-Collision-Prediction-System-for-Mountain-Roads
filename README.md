# AI-Enabled-V2V-V2X-Collision-Prediction-System-for-Mountain-Roads

## Project Overview

This project implements an AI-enabled Vehicle-to-Vehicle (V2V) and Vehicle-to-Everything (V2X) collision prediction system for mountain roads. It integrates three major simulation platforms to create a comprehensive testing environment:

- **SUMO (Simulation of Urban MObility)**: Simulates vehicle movement and traffic scenarios.
- **TensorFlow Lite**: Provides lightweight machine learning models for real-time inference.
- **NS-3 (Network Simulator 3)**: Simulates wireless communication (LTE/5G) between vehicles.

The system predicts potential collisions using a hybrid approach combining physics-based trajectory analysis and a deep learning model, enabling proactive safety measures in complex mountain road environments.

## Features

- **Hybrid Collision Prediction**: Combines trajectory analysis with a deep learning model for accurate risk assessment.
- **Real-time Simulation**: Integrates SUMO, TensorFlow Lite, and NS-3 for end-to-end simulation.
- **Mountain Road Scenarios**: Includes realistic road geometries, speed limits, and traffic patterns specific to mountain roads.
- **Multi-Model Support**: Supports multiple collision prediction models including LSTM, GRU, and Transformer architectures.
- **Performance Monitoring**: Tracks key metrics such as collision rate, prediction accuracy, and communication latency.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.8+**
- **SUMO**: [Download and install SUMO](https://www.eclipse.org/sumo/)
- **TensorFlow**: `pip install tensorflow`
- **NS-3**: [Download and install NS-3](https://www.nsnam.org/)
- **Git**: [Download and install Git](https://git-scm.com/)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd V2V
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

To run the complete simulation with default settings:

```bash
python Mountain_v2v_project/ns-3/ns3_interface.py
```

### Configuration

You can configure simulation parameters in the `config.py` file:

```python
# Simulation settings
SIMULATION_TIME = 240  # seconds
NUM_VEHICLES = 20

# Model settings
MODEL_PATH = "models/best_model.tflite"

# Road network settings
ROAD_NETWORK = "networks/mountain_road.net.xml"
```

### Available Models

The project supports multiple deep learning models. You can switch between them in `config.py`:

- **LSTM**: `MODEL_PATH = "models/best_model.tflite"`
- **GRU**: `MODEL_PATH = "models/gru_model.tflite"`
- **Transformer**: `MODEL_PATH = "models/transformer_model.tflite"`

### Training Models

To train new models, use the training scripts in the `models` directory:

```bash
python models/train_lstm.py
python models/train_gru.py
python models/train_transformer.py
```

## Project Structure

```
V2V/
├── Mountain_v2v_project/
│   ├── ns-3/              # NS-3 simulation scripts
│   ├── sumo/              # SUMO configuration files
│   ├── models/            # Deep learning models and training scripts
│   ├── data/              # Simulation data and logs
│   └── config.py          # Simulation configuration
├── v2v_env/               # Virtual environment
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Testing

To run the test suite:

```bash
python -m unittest test_collision_prediction.py
```

## License

This project is licensed under the terms of the MIT license.
