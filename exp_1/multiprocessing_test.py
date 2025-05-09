import gymnasium as gym
import random
import numpy as np
import ale_py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import multiprocessing as mp
import os
import time
import sys

# Set TensorFlow log level to hide less important messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Force unbuffered stdout for more immediate feedback
sys.stdout.reconfigure(line_buffering=True) if hasattr(
    sys.stdout, "reconfigure"
) else None
print("Starting script...")

# Register Atari environments
gym.register_envs(ale_py)
print("Registered gym environments")


class Agent:
    def __init__(
        self,
        epsilon_decay,
        epsilon_min,
        gamma,
        learning_rate,
        state_size,
        action_size,
        batch_size,
        training_threshold,
    ):
        """CREATING AND DEFINING BASIC PARAMETERS FOR EVALUATION"""
        self.movement_penalty = -1
        # For testing, use a lower epsilon value for more exploitation
        self.epsilon = 0.05  # Lower epsilon for evaluation (even lower than before)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = batch_size
        self.training_threshold = training_threshold

        # Main Model (no target model needed for evaluation)
        self.model = self.create_model()

    def load_weights(self, weights_path):
        """Load model weights from file"""
        try:
            # Load weights from the specified path
            self.model.load_weights(weights_path)
            print(f"Model weights loaded successfully from {weights_path}")
            return True
        except Exception as e:
            print(f"Error loading model weights: {e}")
            import traceback

            traceback.print_exc()
            return False

    def select_action(self, state):
        """DEFINING EPSILON GREEDY STRATEGY FOR EVALUATION"""
        if np.random.rand() > self.epsilon:
            # Use prediction for action selection
            q_value = self.model.predict(state, verbose=0)
            return np.argmax(q_value[0])
        else:
            # Random action with low probability
            return np.random.randint(0, self.action_size)

    def create_model(self):
        """CREATING NEURAL NETWORK"""
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        return model


def run_test_episode(episode_id):
    """Run a single test episode without training or saving"""
    try:
        print(f"Starting test episode {episode_id}")

        # Configuration
        epsilon_decay = 0.999998
        epsilon_min = 0.1
        gamma = 0.99
        learning_rate = 0.000001
        batch_size = 64
        training_threshold = 1000
        weights_path = "./weights/trial_v1_01.weights.h5"

        # Initialize environment
        env = gym.make("ALE/MsPacman-ram-v5")
        state_tuple = env.reset()
        action_size = env.action_space.n
        state_size = env.observation_space.shape[0]

        print(
            f"Episode {episode_id}: Environment initialized. State size: {state_size}, Action size: {action_size}"
        )

        # Initialize agent
        agent = Agent(
            epsilon_decay,
            epsilon_min,
            gamma,
            learning_rate,
            state_size,
            action_size,
            batch_size,
            training_threshold,
        )

        # Load weights
        if not agent.load_weights(weights_path):
            print(f"Episode {episode_id}: Failed to load weights. Aborting.")
            return episode_id, -1

        # Run episode in evaluation mode (no training)
        state = state_tuple[0]
        state = np.reshape(state, [1, state_size])
        done = False
        score = 0

        while not done:
            # Select action (using a very low epsilon for more exploitation)
            agent.epsilon = (
                0.01  # Set to a lower value to make agent more deterministic
            )
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Update state and score (no replay memory updates)
            state = next_state
            score += reward

        # Clean up
        env.close()

        print(f"Episode {episode_id}: Completed. Final score: {score}")

        return episode_id, score

    except Exception as e:
        print(f"Error in episode {episode_id}: {e}")
        import traceback

        traceback.print_exc()
        return episode_id, -1


def main():
    """Main function to run the multiprocessing test without saving"""
    # Number of test episodes
    num_episodes = 100  # Reduced from 2000 to run faster since we're just testing

    # Get number of CPUs for multiprocessing
    num_processes = min(mp.cpu_count(), 8)  # Limit to 8 to avoid overwhelming system
    print(f"Using {num_processes} processes for testing")

    # Run episodes using multiprocessing
    start_time = time.time()
    results = []

    # Create pool and run processes
    with mp.Pool(processes=num_processes) as pool:
        # Process episodes in smaller batches for better management
        batch_size = 20
        for i in range(0, num_episodes, batch_size):
            end_idx = min(i + batch_size, num_episodes)
            print(f"Processing episodes {i} to {end_idx - 1}")

            batch_ids = list(range(i, end_idx))
            batch_results = pool.map(run_test_episode, batch_ids)
            results.extend(batch_results)

            # Report progress
            elapsed = time.time() - start_time
            progress = end_idx / num_episodes
            est_total = elapsed / progress if progress > 0 else 0
            est_remaining = est_total - elapsed
            print(
                f"Progress: {progress * 100:.1f}%, Elapsed: {elapsed:.1f}s, Est. remaining: {est_remaining:.1f}s"
            )

    # Total execution time
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Average time per episode: {elapsed_time / num_episodes:.2f} seconds")

    # Calculate statistics (in memory only, no saving)
    scores = [score for _, score in results if score >= 0]
    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        print(
            f"Results - Avg score: {avg_score:.2f}, Min: {min_score}, Max: {max_score}"
        )
        print(f"Successfully completed {len(scores)} of {num_episodes} episodes")
    else:
        print("No valid scores were obtained")


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method("spawn")
    main()
