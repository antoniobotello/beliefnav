from envs.target_search_env import TargetSearchEnv
import matplotlib.pyplot as plt

env = TargetSearchEnv(grid_size=100, max_steps=70)

obs, info = env.reset(seed=42)

print("Starting simulation...")

plt.ion()
env.render()
plt.pause(1.0)

# Aqui no hay inteligencia todavia, el robot se mueve random
for step in range(50):
    action = env.action_space.sample()  # Genera accion random

    obs, reward, terminated, truncated, info = env.step(
        action
    )  # Recibe accion y acutliza robot_pos

    print(f"Step {step}: reward={reward:.3f}")

    env.render()  # dibuja la nueva posicion

    # This is important
    plt.pause(0.3)

    if terminated or truncated:
        break


print("Finished.")
plt.ioff()
plt.show()
