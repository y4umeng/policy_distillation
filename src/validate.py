import argparse
import gymnasium as gym
import torch
from teacher_network import DQN
from train_teacher import preprocess_env
import ale_py

def test_teacher(model_path, env_name, num_episodes=5):
    """
    Load a trained teacher model and test it on the given environment.
    """
    # Create environment
    env = preprocess_env(env_name)
    num_actions = env.action_space.n

    # Load the trained model
    model = DQN(in_channels=4, num_actions=num_actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Convert the state to a tensor and get the action
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_v)
                action = q_values.argmax(dim=1).item()  # Select action with max Q-value

            # Step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained teacher model on a Gymnasium environment.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file (e.g., teacher_dqn.pth).")
    parser.add_argument("--env-name", type=str, required=True, help="Gymnasium environment name (e.g., ALE/Breakout-v5).")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run for evaluation.")
    args = parser.parse_args()

    gym.register_envs(ale_py)
    test_teacher(args.model_path, args.env_name, args.episodes)
