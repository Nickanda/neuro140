# Deep Reinforcement Learning for PACMAN

This repository contains implementations of various deep reinforcement learning algorithms to train agents to play the classic arcade game PACMAN. The project explores different approaches to Q-learning, with a focus on Categorical Deep Q-Networks (C51) and their application to image-based and RAM-based game states.

## Project Overview

This project explores several reinforcement learning implementations:

1. **Basic DQN Implementation (exp_1)**

   - Standard DQN algorithm applied to RAM state representation
   - Episode-based performance tracking and model weight saving

2. **Categorical Double DQN Implementation (exp_2)**

   - Distributional RL approach
   - Uses a categorical distribution to model expected returns
   - Enhanced stability and performance compared to standard DQN

3. **CNN-based DQN Implementation (cnn_exp1)**

   - Convolutional Neural Network architecture
   - Processes raw game frames as visual input
   - Demonstrates visual feature extraction capabilities

4. **CNN-based Categorical Double DQN (cnn_exp2)**
   - Combines CNN architecture with distributional RL
   - State-of-the-art approach for visual game playing

## Repository Structure

- `analysis/`: Contains analysis scripts for evaluating agent performance

  - `analysis.py`: Performance analysis tools
  - `score_analysis.py`: Scripts for analyzing score progression

- `exp_1/`: Basic DQN implementation with RAM state

  - `final_version_v1.py/ipynb`: DQN implementation code
  - `Deep Q-Learning - PACMAN/`: Saved model weights
  - `images/`: Performance plots
  - `results_tmp/`: Training results data

- `exp_2/`: Categorical DQN implementation with RAM state

  - `final_version_v2.py/ipynb`: C51 algorithm implementation
  - `Deep Q-Learning - PACMAN/`: Saved model weights
  - `images/`: Performance plots
  - `results/`: Training results data

- `cnn_exp1/`: CNN-based DQN implementation

  - `final_version_v1.py/ipynb`: CNN-DQN implementation
  - `Deep Q-Learning - PACMAN/`: Saved model weights

- `cnn_exp2/`: CNN-based Categorical DQN
  - `final_version_v2.py/ipynb`: CNN-C51 implementation
  - `Deep Q-Learning - PACMAN/`: Saved model weights

## Technical Details

### Deep Q-Network (DQN)

The project implements the DQN algorithm with the following key features:

- Experience replay memory to store and sample state transitions
- Target network for stable Q-value estimation
- Epsilon-greedy exploration strategy
- Periodic model weight saving

### Categorical DQN (C51)

The distributional RL approach offers several advantages:

- Models full distribution of possible returns instead of just the mean
- Uses 51 atoms to represent the value distribution
- Implements distributional Bellman update with projection onto fixed support
- Provides more stable learning and better performance

### Double DQN

The Double DQN approach addresses the overestimation bias in standard DQN:

- Uses two separate networks (main and target) for action selection and evaluation
- Main network selects the best action for the next state
- Target network evaluates the Q-value of that action
- Decouples action selection from action evaluation
- Reduces overoptimistic value estimates
- Results in more stable and reliable learning

### CNN Architecture

The convolutional networks process raw game frames with:

- Image preprocessing (grayscale conversion, downsampling)
- Multiple convolutional layers for feature extraction
- Dense layers for Q-value or distribution prediction

## Environment

The project uses the OpenAI Gymnasium environment with the Arcade Learning Environment (ALE):

- `ALE/MsPacman-v5`: Image-based environment for CNN approaches
- `ALE/MsPacman-ram-v5`: RAM state representation (128-dimensional vector)

## Key Training Parameters

- Episodes: 2100
- Epsilon decay: 0.999998
- Minimum epsilon: 0.1
- Discount factor (gamma): 0.99
- Learning rate: 0.000001
- Batch size: 64
- Training threshold: 1000
- Target network update frequency: 10

## Results

Training results are saved as:

1. Model weights (`.h5` files)
2. Performance plots (`.png` files)
3. Raw score data (`.txt` files)

The performance plots show the progression of scores across episodes, demonstrating the learning capability of each algorithm.

## Getting Started

To run the experiments, you need:

1. Python 3.x
2. TensorFlow
3. Gymnasium with ALE
4. NumPy, Matplotlib

You can run the different experiments using either the Python scripts or Jupyter notebooks provided in each experiment directory.
