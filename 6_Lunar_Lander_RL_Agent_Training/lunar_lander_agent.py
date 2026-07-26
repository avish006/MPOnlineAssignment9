import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

print("Creating LunarLander environment...")
env = gym.make("LunarLander-v3")

print("Initializing PPO agent...")
model = PPO("MlpPolicy", env, verbose=1)

print("Training agent for 50,000 timesteps (might take a few minutes)...")
model.learn(total_timesteps=50000)

print("Saving model...")
model.save("ppo_lunar_lander")

print("Evaluating agent...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

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
