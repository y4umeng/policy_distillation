import argparse
import gymnasium as gym
import torch
from train_teacher import preprocess_env
import ale_py
from load_teacher import get_qnet
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams

def test_teacher(model_path, env_name, num_episodes=5, seed=121):
    """
    Load a trained teacher model and test it on the given environment.
    """
    # Create environment


    hyperparams, stats_path = get_saved_hyperparams("rl-trained-agents/dqn/" + env_name + "/config.yml")
    
    env = create_test_env(
        env_name,
        stats_path=stats_path,
        seed=seed,
        should_render=False,
        hyperparams=hyperparams,
        #env_kwargs=env_kwargs,
    )

    # old env
    env = preprocess_env(env_name)

    num_actions = env.action_space.n

    # Load the trained model
    # model = DQN(in_channels=4, num_actions=num_actions)
    # model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model = get_qnet()
    model.eval()  # Set the model to evaluation mode

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Convert the state to a tensor and get the action
            state_v = torch.tensor(state, dtype=torch.float32, device="cuda").squeeze().unsqueeze(0)
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
    parser.add_argument("--model-path", type=str, required=False, help="Path to the trained model file (e.g., teacher_dqn.pth).")
    parser.add_argument("--env-name", type=str, default="BreakoutNoFrameskip-v4", help="Gymnasium environment name (e.g., ALE/Breakout-v5).")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run for evaluation.")
    args = parser.parse_args()

    # gym.register_envs(ale_py)
    test_teacher(args.model_path, args.env_name, args.episodes)
