import gymnasium as gym
import highway_env
for i in range(10):
    # env = gym.make('safe-highway-fast-v0', render_mode='human')
    env = gym.make('safe-intersection-v0', render_mode='human')

    obs, info = env.reset()
    # 用于存储每一步的 is_first 状态
    is_first_history = []

    done = truncated = False
    i = 0
    while not (done or truncated):
        action = -0.1 # Your agent code here
        # action = [0.1, 0.1]  # Your agent code here
        i += 1
        # if i%2 == 0:
        #     action[1] = -0.1
        obs, reward, done, truncated, info = env.step(action)
        # 保存当前时间步的 is_first 状态
        is_first_history.append(info.get("is_first", False))  # 默认False，如果没有这个key的话
        # results = env.step(action)
        print(1111)
# import safety_gymnasium
#
# env_id = 'SafetyPointGoal1-v0'
# env = safety_gymnasium.make(env_id,render_mode='human',camera_name='fixedfar', width=1024, height=1024)
#
# obs, info = env.reset()
# while True:
#     act = env.action_space.sample()
#     obs, reward, cost, terminated, truncated, info = env.step(act)
#     if terminated or truncated:
#         break
#     env.render()