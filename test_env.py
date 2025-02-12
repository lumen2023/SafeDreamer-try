import gymnasium as gym
import highway_env
# env = gym.make('safe-highway-fast-v0', render_mode='human')
env = gym.make('safe-intersection-v0', render_mode='human')

obs, info = env.reset()
# 用于存储每一步的 is_first 状态
is_first_history = []

done = truncated = False
while not (done or truncated):
    action = 1.5 # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
    # 保存当前时间步的 is_first 状态
    is_first_history.append(info.get("is_first", False))  # 默认False，如果没有这个key的话

    results = env.step(action)
    print(1111)