import functools

import embodied
import numpy as np

import gymnasium as gym

class FromGymnasium(embodied.Env):

  def __init__(self, env, obs_key='observation', act_key='action', **kwargs):
    # 不必要的参数platform
    if 'platform' in kwargs:
      del kwargs['platform']  # 删除 platform 参数
    if isinstance(env, str):
      # self._env = gym.make(env, render_mode='human') # render_mode='human'
      # self._env = gym.make(env, **kwargs)  # render_mode='human'
      self._env = gym.make(env) # render_mode='human'
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    # self.speed = 0
    self.cost = 0
    self.cost_vases_contact = 0
    self.cost_vases_velocity = 0
    self.cost_hazards = 0
    self.cost_gremlins = 0
  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'cost': embodied.Space(np.float32),
        # 'speed': embodied.Space(np.float32),
        'crash': embodied.Space(bool),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  @property
  def task(self):
    return self._env.task


  def initial_reset(self):
    obs, info = self._env.reset(seed=0)
    self._done = True
    self._info = None
    return self._obs(obs, 0.0, 0.0,0,False, is_first=True)

  def step(self, action):
    """
    执行环境的一个步骤。

    如果需要重置环境或上一个步骤已完成，则先重置环境。
    根据行动字典或键来调整动作。
    应用动作并累积相应的成本。
    当环境完成时，重置成本。
    返回观察结果、奖励、成本以及是否完成的标志。
    """
    # print("!!!!!!!!!!!!!!")
    # 检查是否需要重置环境或上一个步骤是否已完成
    if action['reset'] or self._done:
      self._done = False
      obs, info = self._env.reset()
      return self._obs(obs, 0.0, 0.0, 0, False,is_first=True)

    # 根据配置调整动作输入
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]

    # 以下代码被注释掉，可能是为了允许负值的动作输入
    # if action[0] < 0.0:
      # action[0] = 0.0

    # # 执行环境步骤并获取结果
    # obs, reward, cost, terminated, truncated, self._info = self._env.step(action)
    # 执行环境步骤并处理返回值
    result = self._env.step(action)
    if len(result) == 6:
      obs, reward, cost, terminated, truncated, self._info = result
    elif len(result) == 5:
      obs, reward, terminated, truncated, self._info = result
      cost = self._info['cost']
      speed = self._info['speed']
      crash = self._info['crashed']
      if self._info['crashed'] == True or self._info['on_road'] == False:
        # print('crashed')
        crash = True
      # out = self._info['on']
    # 累积不同类型的接触成本
    if 'cost_vases_contact' in self._info.keys():
      self.cost_vases_contact += self._info['cost_vases_contact']
    if 'cost_vases_velocity' in self._info.keys():
      self.cost_vases_velocity += self._info['cost_vases_velocity']
    if 'cost_hazards' in self._info.keys():
      self.cost_hazards += self._info['cost_hazards']
    if 'cost_gremlins' in self._info.keys():
      self.cost_gremlins += self._info['cost_gremlins']

    # 累积总成本
    self.cost += cost

    # 检查环境是否完成
    self._done = terminated or truncated

    # 如果环境完成，重置所有成本
    if self._done:
      self.cost = 0
      # self.spped = 0
      self.cost_vases_contact = 0
      self.cost_vases_velocity = 0
      self.cost_hazards = 0
      self.cost_gremlins = 0
    # 返回观察结果、奖励、成本以及是否完成的标志
    return self._obs(
        obs, reward, cost, speed, crash,
        # obs, reward, cost, crash, self._info,
        # obs, reward, cost, crash,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
          self, obs, reward, cost, speed, crash, is_first=False, is_last=False, is_terminal=False):
    # self, obs, reward, cost, crash, info, is_first=False, is_last=False, is_terminal=False):
    """
    处理观察(observations)，将接收到的观察数据以及其他相关信息整合为一个结构化的字典。

    参数:
    - obs: 当前的观察数据，可以是任意类型，但函数会将其处理为字典形式。
    - reward: 当前步获得的奖励值。
    - cost: 当前步产生的成本或消耗。
    - is_first: 是否为序列中的第一个观察，默认为False。
    - is_last: 是否为序列中的最后一个观察，默认为False。
    - is_terminal: 是否为一个终端状态的观察，默认为False。

    返回:
    - 一个字典，包含了处理后的观察数据以及附加的信息（奖励、成本、是否为序列起始/结束/终端状态）。
    """
    # 如果观察字典为空，则将obs包装成一个字典，使用_obs_key作为键
    if not self._obs_dict:
      obs = {self._obs_key: obs}

    # 将观察数据结构展平，以便于后续处理
    obs = self._flatten(obs)

    # 确保观察字典中的每个值都被转换为numpy数组
    obs = {k: np.asarray(v) for k, v in obs.items()}

    # 更新观察字典，加入奖励、成本及序列标记信息
    obs.update(
      reward=np.float32(reward),
      cost=np.float32(cost),
      speed=np.float32(speed),
      crash=crash,
      is_first=is_first,
      is_last=is_last,
      is_terminal=is_terminal)

    # 返回更新后的观察字典
    return obs

  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
