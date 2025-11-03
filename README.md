# Vertical Federated Learning (VFL) Simulation

This repository contains a proof-of-concept simulation of **Vertical Federated Learning (VFL)** for training a machine learning model across two organizations without sharing raw data.

## Overview

This simulation demonstrates how two organizations can collaboratively train a loan eligibility prediction model:
- **Guest (Organization A)**: Hamrah-e Avval (Telecom Company) - provides telecom usage data
- **Host (Organization B)**: Bank - provides financial data and holds the training labels

The key principle of VFL is that **raw data never leaves each organization's premises**. Instead, only encrypted intermediate computations are shared, allowing the model to learn from combined features while preserving data privacy.

## What is Vertical Federated Learning?

In **Vertical Federated Learning (VFL)**, different organizations have different features about the same set of users/samples. This contrasts with Horizontal Federated Learning (HFL), where organizations have the same features but different samples.

**Example Scenario:**
- A telecom company knows: data usage, charge frequency
- A bank knows: account balance, loan history
- Both know about the same users (same user IDs)
- Goal: Predict loan eligibility using features from both organizations

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Guest (Telecom)   │         │    Host (Bank)      │
│                     │         │                     │
│  - Data Usage       │         │  - Account Balance  │
│  - Charge Freq      │         │  - Loan History     │
│                     │         │  - Labels (y)       │
│  Weights: W_a       │         │  Weights: W_b       │
└──────────┬──────────┘         └──────────┬──────────┘
           │                                │
           │  z_a = X_a · W_a               │
           │───────────────┐                │
           │               │                │
           │               ▼                │
           │        [Coordinator]           │
           │               │                │
           │               │ z_total        │
           │               │───────────────►│
           │               │                │  z_b = X_b · W_b
           │               │                │  Loss & Gradients
           │               │                │
           │◄──────────────┼────────────────│
           │  common_grad  │                │
           │               │                │
```

## Components

### 1. Data Generation (`generate_synthetic_data`)
- Creates synthetic datasets for both organizations
- Simulates 1000 common users
- Loan eligibility is determined by a hidden formula combining features from both organizations
- This hidden logic is what the model must learn

**Guest Data:**
- `user_id`: Common identifier
- `data_usage_gb`: Monthly data usage (GB)
- `charge_frequency`: Number of charges per month

**Host Data:**
- `user_id`: Common identifier
- `account_balance`: Account balance
- `loan_history_score`: Credit history score (0-1)
- `is_eligible_for_loan`: Binary label (target)

### 2. Blockchain Simulator (`BlockchainSimulator`)
- Simulates event logging to a consortium blockchain
- Provides an audit trail of all VFL operations
- Logs events like:
  - System initialization
  - Data preparation
  - Training start/end
  - Epoch completion
- In Phase 3, this will be replaced with real smart contract interactions

### 3. VFL Guest (`VFLGuest`)
- Represents Organization A (Telecom)
- Responsibilities:
  - Preprocesses local data (normalization, bias term)
  - Computes partial prediction: `z_a = X_a · W_a`
  - Sends encrypted `z_a` to coordinator
  - Receives gradient and updates local weights `W_a`

### 4. VFL Host (`VFLHost`)
- Represents Organization B (Bank)
- Responsibilities:
  - Preprocesses local data
  - Holds training labels (`y`)
  - Receives `z_a` from guest
  - Computes `z_b = X_b · W_b`
  - Aggregates: `z_total = z_a + z_b`
  - Computes predictions, loss, and gradients
  - Updates local weights `W_b`
  - Sends gradient back to guest

### 5. Training Loop (`main_training_loop`)
Orchestrates the VFL training process:
1. Initialize blockchain and generate data
2. Initialize Guest and Host
3. For each epoch:
   - Guest computes and sends `z_a`
   - Host computes `z_b`, aggregates to get `z_total`
   - Host computes loss and gradients
   - Gradients are sent back to Guest
   - Both parties update their weights
   - Events are logged to blockchain
4. Display final results and audit trail

## How VFL Works

### Forward Pass
1. **Guest** computes partial prediction: `z_a = X_a · W_a`
2. **Host** computes partial prediction: `z_b = X_b · W_b`
3. **Coordinator** securely aggregates: `z_total = z_a + z_b`
4. **Host** computes final prediction: `ŷ = sigmoid(z_total)`

### Backward Pass
1. **Host** computes loss: `L = -mean(y·log(ŷ) + (1-y)·log(1-ŷ))`
2. **Host** computes common gradient: `∂L/∂z = (ŷ - y) / N`
3. **Host** computes its gradient: `∂L/∂W_b = X_b^T · ∂L/∂z`
4. **Host** sends `∂L/∂z` to Guest (via coordinator)
5. **Guest** computes its gradient: `∂L/∂W_a = X_a^T · ∂L/∂z`
6. Both parties update weights using gradient descent

### Security Aspects (Noted in Code)
- In real implementations:
  - `z_a` is encrypted before transmission
  - Gradients are encrypted during transmission
  - Homomorphic encryption or secure multiparty computation is used
  - A coordinator handles secure aggregation

## Usage

### Python Script
```bash
python vfl_simulation.py
```

### Jupyter Notebook
```bash
jupyter notebook vfl_simulation.ipynb
```

The simulation runs for 15 epochs by default and displays:
- Data generation progress
- Initialization messages
- Loss at each epoch
- Final training results
- Complete blockchain audit trail

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation

Install dependencies:
```bash
pip install numpy pandas
```

## Key Features

1. **Privacy-Preserving**: Raw data never leaves each organization
2. **Collaborative Learning**: Model learns from combined features across organizations
3. **Audit Trail**: All events logged to blockchain (simulated)
4. **Simple Implementation**: Easy to understand VFL basics
5. **Extensible**: Can be extended with real encryption and blockchain integration

## Limitations of This Simulation

This is a **proof-of-concept** simulation. Real-world implementations would include:

1. **Cryptography**: Real encryption for intermediate values
2. **Secure Aggregation**: Homomorphic encryption or secure multiparty computation
3. **Coordinator**: Actual coordinator service for secure aggregation
4. **Blockchain**: Real smart contract integration
5. **User Alignment**: Secure user ID matching (Private Set Intersection)
6. **Security Protocols**: Protection against various attacks (inference attacks, gradient leakage, etc.)

## Future Work (Phase 3)

- Replace blockchain simulator with real smart contract calls
- Add real encryption mechanisms
- Implement secure coordinator
- Add comprehensive security testing
- Performance optimization

## File Structure

```
.
├── vfl_simulation.py      # Main Python script
├── vfl_simulation.ipynb   # Jupyter notebook version
├── README.md              # This file
└── venv/                  # Virtual environment (optional)
```

## References

This simulation is part of a thesis project on Vertical Federated Learning. For more details on VFL, see:
- Vertical Federated Learning frameworks (e.g., FATE, TensorFlow Federated)
- Secure multiparty computation
- Homomorphic encryption
