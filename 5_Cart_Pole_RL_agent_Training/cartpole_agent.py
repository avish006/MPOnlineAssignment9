import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

print("Creating CartPole environment...")
env = gym.make("CartPole-v1")

print("Initializing PPO agent...")
# PPO is a popular RL algorithm
model = PPO("MlpPolicy", env, verbose=1)

print("Training agent for 20,000 timesteps...")
model.learn(total_timesteps=20000)

print("Saving model...")
model.save("ppo_cartpole")

# Evaluate the trained agent
print("Evaluating agent...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Optional: Run a test episode to show it works
obs, info = env.reset()
total_reward = 0
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
        
print(f"Test episode reward: {total_reward}")
env.close()
