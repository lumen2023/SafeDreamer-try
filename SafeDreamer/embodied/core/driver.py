import collections

import numpy as np

from .basics import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0, lag=0.0, lag_p=0.0, lag_i=0.0, lag_d=0.0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode, lag, lag_p, lag_i, lag_d)

  def _step(self, policy, step, episode, lag, lag_p, lag_i, lag_d):
    """
    执行环境的一个步骤。

    该函数负责与环境进行一次交互，应用给定的策略来选择动作，并处理观察结果和奖励。
    它还负责在内部跟踪每个环境的episode，并在episode结束时调用相应的回调函数。

    参数:
    - policy: 一个函数，用于选择要采取的动作。
    - step: 当前的步骤数。
    - episode: 当前完成的episode数量。
    - lag: 滞后项系数，用于计算lagrange惩罚。
    - lag_p: 滞后项系数，针对lagrange惩罚的P部分。
    - lag_i: 滞后项系数，针对lagrange惩罚的I部分。
    - lag_d: 滞后项系数，针对lagrange惩罚的D部分。

    返回:
    - step: 更新后的步骤数。
    - episode: 更新后的完成的episode数量。
    """
    # 确保所有动作的长度与环境数量一致
    assert all(len(x) == len(self._env) for x in self._acts.values())

    # 过滤掉日志相关的动作，因为它们不直接影响环境
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}

    # 执行动作并获取环境的观察结果
    obs = self._env.step(acts)
    # speed = obs["info"][0].get('speed', 0)
    ####LYZ-物理信息记录###

    # 添加lagrange惩罚和控制项到观察结果中
    obs['speed'] = obs["speed"] * np.ones(len(self._env))
    obs['lagrange_penalty'] = lag * np.ones(len(self._env))
    obs['lagrange_p'] = lag_p * np.ones(len(self._env))
    obs['lagrange_i'] = lag_i * np.ones(len(self._env))
    obs['lagrange_d'] = lag_d * np.ones(len(self._env))

    # 将观察结果中的每个值通过convert函数转换
    obs = {k: convert(v) for k, v in obs.items()}

    # 确保所有观察结果的长度与环境数量一致
    assert all(len(x) == len(self._env) for x in obs.values()), obs

    # 使用当前策略选择下一步的动作
    acts, self._state = policy(obs, self._state, **self._kwargs)

    # 将选定的动作通过convert函数转换
    acts = {k: convert(v) for k, v in acts.items()}

    # 如果任何环境的episode在此步骤结束，则应用mask来重置相应的状态
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}

    # 准备重置信号和更新内部动作
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts

    # 合并观察结果和动作，形成transition
    trns = {**obs, **acts}

    # 如果任何环境的episode在此步骤开始，则清除相应的episode历史记录
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()

    # 对于每个环境，将当前步骤的transition添加到其episode历史记录中，并调用步骤回调函数
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]

    # 更新步骤数
    step += 1

    # 如果任何环境的episode在此步骤结束，则调用episode回调函数
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1

    # 返回更新后的步骤数和完成的episode数量
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
