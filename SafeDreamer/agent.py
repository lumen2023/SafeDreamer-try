import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj

def symexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

# 代码定义了一个名为 Agent 的类，继承自 nj.Module，主要用于强化学习中的代理（Agent）部分。
# 使用了类似于强化学习中常见的世界模型（World Model）、任务行为（Task Behavior）以及探索行为（Exploration Behavior）的结构，
# 结合了行为规划算法（如CEMPlanner、PIDPlanner等）

@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    elif config.expl_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.task_behavior.ac, self.wm, self.act_space, self.config, name='expl_behavior')
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  # 此方法用于初始化代理的策略。它会调用世界模型、任务行为和探索行为的初始化方法，
  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  # 此方法仅初始化世界模型，用于训练时的初始状态
  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  # 该方法定义了代理的行为策略，接受以下参数：
  def policy(self, obs, state, mode='train'):
    #
    # obs: 当前的环境观察。
    # state: 当前的状态，包括前一个时间步的潜在状态、任务状态和探索状态。
    # mode: 模式
    # 'train'（训练模式）、'eval'（评估模式）或'explore'（探索模式）

    # 根据mode，代理会采取不同的行为：
    # 训练模式(train)：代理根据当前的状态和观察来计算一个动作，并返回相应的输出。
    # 评估模式(eval)：用于评估时，代理返回的输出不会有探索行为的干扰。
    # 探索模式(explore)：代理使用探索行为（例如PID规划器）来进行更加随机的探索。

    print("进入Agent policy----")
    self.config.jax.jit and print('Tracing policy function.')

    # 对观测进行预处理
    obs = self.preprocess(obs)

    # 解构状态，获取先前的潜在状态、动作，以及任务和探索状态
    (prev_latent, prev_action), task_state, expl_state = state

    # 如果使用PID控制器作为探索行为，更新探索和任务状态中的PID参数。
    if self.config.expl_behavior in ['PIDPlanner']:
      expl_state['lagrange_penalty'] = obs['lagrange_penalty']
      task_state['lagrange_penalty'] = obs['lagrange_penalty']
      expl_state['lagrange_p'] = obs['lagrange_p']
      task_state['lagrange_p'] = obs['lagrange_p']
      expl_state['lagrange_i'] = obs['lagrange_i']
      task_state['lagrange_i'] = obs['lagrange_i']
      expl_state['lagrange_d'] = obs['lagrange_d']
      task_state['lagrange_d'] = obs['lagrange_d']

    # 使用编码器对观测进行编码，得到嵌入表示。
    embed = self.wm.encoder(obs)

    # 使用RSSM进行一步观测更新，得到新的潜在状态。
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    #self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)

    # 根据模式选择策略输出。
    if mode == 'eval':
      if self.config.expl_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
        outs = expl_outs
        outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
      else:
        outs = task_outs
        outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'explore':
      outs = expl_outs
      if self.config.expl_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
        outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
      else:
        outs['log_entropy'] = outs['action'].entropy()
        outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      if self.config.task_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
        outs = expl_outs
        outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
      else:
        outs = task_outs
        outs['log_entropy'] = outs['action'].entropy()

    # 对于非规划类探索行为，初始化一些日志变量
    if self.config.expl_behavior not in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
      outs['log_plan_action_mean'] = jnp.zeros(outs['action'].shape)
      outs['log_plan_action_std'] = jnp.zeros(outs['action'].shape)
      outs['log_plan_num_safe_traj'] = jnp.zeros(outs['action'].shape[:1])
      outs['log_plan_ret'] = jnp.zeros(outs['action'].shape[:1])
      outs['log_plan_cost'] = jnp.zeros(outs['action'].shape[:1])

    # 对于PID控制器探索行为，记录PID参数。
    if self.config.expl_behavior in ['PIDPlanner']:
      outs['log_lagrange_penalty'] = obs['lagrange_penalty'] * jnp.ones(outs['action'].shape[:1])
      outs['log_lagrange_p'] = obs['lagrange_p'] * jnp.ones(outs['action'].shape[:1])
      outs['log_lagrange_i'] = obs['lagrange_i'] * jnp.ones(outs['action'].shape[:1])
      outs['log_lagrange_d'] = obs['lagrange_d'] * jnp.ones(outs['action'].shape[:1])

    # 更新状态，包括当前的潜在状态、选定的动作、任务状态和探索状态。
    state = ((latent, outs['action']), task_state, expl_state)

    # 返回策略输出和更新后的状态。
    return outs, state

  # 该方法用于训练代理：
  def train(self, data, state):
    """
    训练方法。通过调用世界模型和任务行为的训练方法来更新模型，并返回训练后的输出和状态。

    参数:
    - data: 包含训练数据。
    - state: 当前的状态。

    返回:
    - outs: 训练后的输出。
    - state: 更新后的状态。
    - metrics: 训练过程中的各种度量。
    """
    print("进入Agent train----")
    # 如果配置了JAX的即时编译(jit)，则打印训练函数被追踪的消息。
    self.config.jax.jit and print('Tracing train function.')

    # 初始化度量字典，用于收集训练过程中的各种度量。
    metrics = {}

    # 对输入数据进行预处理。
    data = self.preprocess(data)

    # 调用世界模型的训练方法，更新状态，并收集世界模型的输出和度量。
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)

    # 构建上下文，包括原始数据和世界模型的后验(posterior)信息。
    context = {**data, **wm_outs['post']}

    # 将上下文中的每个元素重新整形，以便后续处理。
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)

    # 调用任务行为的训练方法，收集训练输出和度量，并更新度量字典。
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)

    # 如果配置了探索行为且不是特定的规划器类型，则也对探索行为进行训练，并更新度量字典。
    if self.config.expl_behavior != 'None' and self.config.expl_behavior not in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})

    # 初始化输出字典，准备返回。
    outs = {}

    # 返回训练后的输出、状态和度量。
    return outs, state, metrics


  def report(self, data):
      """
      生成环境报告。
      根据输入的数据，预处理后生成一个综合报告，该报告包含了世界模型、任务行为和探索行为的度量。
      参数:
      - data: 输入数据，用于生成报告。
      返回值:
      - report: 一个字典，包含了所有度量的综合报告。
      """
      # 如果配置了JAX的JIT编译，则打印追踪报告函数的消息
      self.config.jax.jit and print('Tracing report function.')

      # 对输入数据进行预处理
      data = self.preprocess(data)

      # 初始化报告字典
      report = {}

      # 更新报告字典，添加世界模型的度量
      report.update(self.wm.report(data))

      # 获取任务行为的度量，并以'task_'为前缀添加到报告中
      mets = self.task_behavior.report(data)
      report.update({f'task_{k}': v for k, v in mets.items()})

      # 如果探索行为和任务行为不同，则获取探索行为的度量，并以'expl_'为前缀添加到报告中
      if self.expl_behavior is not self.task_behavior:
        mets = self.expl_behavior.report(data)
        report.update({f'expl_{k}': v for k, v in mets.items()})

      # 返回综合报告
      return report

  def report_eval(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report_eval(data))
    return report



  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
      """
      初始化模型。

      参数:
      obs_space: 观测空间的字典，包含了不同类型的观测数据及其形状。
      act_space: 动作空间的字典，包含了动作数据及其形状。
      config: 配置参数的字典，包含了模型的各种超参数和设置。
      """
      # 存储观测空间、动作空间和配置参数
      self.obs_space = obs_space
      self.act_space = act_space['action']
      self.config = config

      # 计算观测空间中各部分的形状，忽略日志相关的观测
      shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
      shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}

      # 初始化编码器，用于将观测数据编码为模型可以处理的表示
      self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')

      # 初始化RSSM（递归状态空间模型），用于建模环境的动态特性
      self.rssm = nets.RSSM(**config.rssm, name='rssm')

      # 初始化多个预测头：解码器、奖励预测、连续性预测
      self.heads = {
          'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
          'reward': nets.MLP((), **config.reward_head, name='rew'),
          'cont': nets.MLP((), **config.cont_head, name='cont')}

      # 如果配置中包含使用成本，则添加成本预测头
      if self.config.use_cost:
        self.heads['cost'] = nets.MLP((), **config.cost_head, name='cost')

      # 初始化优化器，用于模型参数的更新
      self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)

      # 处理损失函数的权重，区分图像和向量数据
      scales = self.config.loss_scales.copy()
      image, vector = scales.pop('image'), scales.pop('vector')
      scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
      scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
      self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
      """
      训练模型的函数。

      参数:
      - data: 训练数据。
      - state: 模型的初始状态。

      返回:
      - state: 更新后的模型状态。
      - outs: 模型的输出。
      - metrics: 训练过程中的度量指标。
      """
      print("\n***世界模型模型训练***\n")
      # 初始化模块列表，包括编码器、RSSM模型和所有头部网络
      modules = [self.encoder, self.rssm, *self.heads.values()]

      # 使用优化器对模块进行训练，同时计算损失和度量指标
      mets, (state, outs, metrics) = self.opt(
          modules, self.loss, data, state, has_aux=True)

      # 更新度量指标，将优化过程中的度量指标添加到metrics中
      metrics.update(mets)

      # 返回更新后的状态、模型输出和度量指标
      return state, outs, metrics

  # 计算模型损失
  def loss(self, data, state):
    # 使用编码器对数据进行嵌入
    embed = self.encoder(data)
    # 解包前一个状态
    prev_latent, prev_action = state
    # 准备先前的动作序列
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    # 使用RSSM观察模型处理嵌入和动作，得到后验和先验分布
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    # 初始化分布字典
    dists = {}
    # 构建特征集合，包括后验分布和嵌入
    feats = {**post, 'embed': embed}
    # 遍历所有头部，计算各个任务的分布
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    # 初始化损失字典
    losses = {}
    # 计算动态损失和表示损失
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    # 遍历所有分布，计算各项损失
    for key, dist in dists.items():
      if key == 'cost':
        # 对于成本分布，使用条件损失
        condition = jnp.greater_equal(data['cost'], 1.0)
        loss = -dist.log_prob(data['cost'].astype(jnp.float32))
        loss = jnp.where(condition, self.config.cost_weight * loss, loss)
      else:
        # 对于其他分布，直接计算负对数概率作为损失
        loss = -dist.log_prob(data[key].astype(jnp.float32))
      # 确保损失形状正确
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      # 更新损失字典
      losses[key] = loss
    # 对所有损失进行缩放
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    # 计算总模型损失
    model_loss = sum(scaled.values())
    # 准备输出字典
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    # 提取最后一个时间步的潜在状态和动作
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    # 更新状态
    state = last_latent, last_action
    # 计算并收集评估指标
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    print("\n***世界模型loss获取***\n")
    # 返回平均模型损失、新状态、输出字典和评估指标
    return model_loss.mean(), (state, out, metrics)


  def new_imagine(self, policy, start, horizon, use_planner=False):
      """
      根据给定的策略和起始状态，想象一系列的行动轨迹。

      参数:
      - policy: 决策策略，用于选择行动。
      - start: 起始状态，包含初始信息。
      - horizon: 视野范围，即想象的步数。
      - use_planner: 是否使用规划器，以影响行动的选择过程。

      返回:
      - traj: 行动轨迹，包含每个时间步的状态和行动。
      """
      # 计算起始状态的持续概率，用于后续计算
      first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
      # 获取初始状态的键值
      keys = list(self.rssm.initial(1).keys())

      if use_planner:
          # 如果使用规划器，添加额外的键值，以处理更复杂的策略
          keys += ['action_mean','action_std','action_mean_expl','action_std_expl', 'lagrange_multiplier', 'penalty_multiplier']
          # 过滤start，只保留需要的键值，并添加当前策略选择的行动
          start = {k: v for k, v in start.items() if k in keys}
          start['action'] = policy(start,0)
          # 定义每一步的处理函数，考虑当前视野范围
          def step(prev, current_horizon):
              prev = prev.copy()
              action_mean = prev['action_mean_expl']
              action_std = prev['action_std_expl']
              state = self.rssm.img_step(prev, prev.pop('action'))
              return {**state, 'action_mean_expl':action_mean, 'action_std_expl':action_std, 'action': policy(prev,current_horizon+1)}
      else:
          # 如果不使用规划器，直接根据策略选择行动
          start = {k: v for k, v in start.items() if k in keys}
          start['action'] = policy(start)
          # 定义每一步的处理函数，不考虑视野范围
          def step(prev, _):
              prev = prev.copy()
              state = self.rssm.img_step(prev, prev.pop('action'))
              return {**state, 'action': policy(state)}

      # 使用scan函数应用step函数，生成轨迹
      traj = jaxutils.scan(
          step, jnp.arange(horizon), start, self.config.imag_unroll)
      # 添加起始状态到轨迹中
      traj = {
          k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}

      # 计算轨迹中每一步的持续概率
      cont = self.heads['cont'](traj).mode()
      traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)

      # 计算折扣因子，用于后续计算权重
      discount = 1 - 1 / self.config.horizon
      traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount

      return traj
  def imagine(self, policy, start, horizon, use_planner=False):
      first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
      keys = list(self.rssm.initial(1).keys())
      if use_planner:
          keys += ['action_mean', 'action_std', 'lagrange_multiplier', 'penalty_multiplier']
          start = {k: v for k, v in start.items() if k in keys}
          start['action'] = policy(start, 0)

          def step(prev, current_horizon):  # add the current_horizon
              prev = prev.copy()
              action_mean = prev['action_mean']
              action_std = prev['action_std']
              state = self.rssm.img_step(prev, prev.pop('action'))
              return {**state, 'action_mean': action_mean, 'action_std': action_std,
                      'action': policy(prev, current_horizon + 1)}
      else:
          start = {k: v for k, v in start.items() if k in keys}
          start['action'] = policy(start)

          def step(prev, _):
              prev = prev.copy()
              state = self.rssm.img_step(prev, prev.pop('action'))
              return {**state, 'action': policy(state)}
      traj = jaxutils.scan(
          step, jnp.arange(horizon), start, self.config.imag_unroll)
      traj = {
          k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
      cont = self.heads['cont'](traj).mode()
      traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
      discount = 1 - 1 / self.config.horizon
      traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
      return traj
  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def report_eval(self, data_expand):
    state = self.initial(len(data_expand['is_first']))
    report = {}
    report.update(self.loss(data_expand, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data_expand)[:, :5], data_expand['action'][:, :5],
        data_expand['is_first'][:, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data_expand['action'][:, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      report[f'openl_{key}'] = jaxutils.video_grid(model)
    for key in self.heads['decoder'].mlp_shapes.keys():
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      report[f'openl_{key}'] = model
      if 'openl_observation' in report.keys() and not self.config.use_cost:
        report[f'openl_cost'] = self.cost_from_recon(report['openl_observation'])
    return report

  def cost_from_recon(self, recon):
    print("世界模型--重建模型--cost")
    # jax format
    hazards_size = 0.25
    batch_size = recon.shape[0] * recon.shape[1]
    hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
    hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
        batch_size,
        -1,
    )

    condition = jnp.less_equal(hazards_dist, hazards_size)
    cost = jnp.where(condition, 1.0, 0.0)
    cost = cost.sum(1)
    condition = jnp.greater_equal(cost, 1.0)
    cost = jnp.where(condition, 1.0, 0.0)

    cost = cost.reshape(recon.shape[0], recon.shape[1])
    return cost



  def _metrics(self, data, dists, post, prior, losses, model_loss):
      """
      计算并汇总各种性能指标。

      参数:
      - data: 真实数据集，包含实际观测值和其他相关信息。
      - dists: 模型预测的分布，用于计算预测值和实际值之间的差异。
      - post: 后验分布，基于观测数据得到的特征分布。
      - prior: 先验分布，未考虑当前观测数据时的特征分布。
      - losses: 损失字典，包含各种损失项。
      - model_loss: 模型损失，用于评估模型的整体性能。

      返回:
      - metrics: 性能指标字典，包含各种计算得到的统计量。
      """

      # 定义熵计算函数，用于后续计算先验和后验的熵。
      entropy = lambda feat: self.rssm.get_dist(feat).entropy()
      metrics = {}

      # 计算并记录先验和后验分布的熵的统计信息。
      metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
      metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))

      # 计算并记录各项损失的平均值和标准差。
      metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
      metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})

      # 计算并记录模型损失的平均值和标准差。
      metrics['model_loss_mean'] = model_loss.mean()
      metrics['model_loss_std'] = model_loss.std()

      # 计算并记录奖励的预测和实际值的最大绝对值。
      metrics['reward_max_data'] = jnp.abs(data['reward']).max()
      metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()

      # 如果存在奖励预测，且不处于调试模式下，计算并记录奖励的平衡统计信息。
      if 'reward' in dists and not self.config.jax.debug_nans:
        stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
        metrics.update({f'reward_{k}': v for k, v in stats.items()})

      # 如果存在继续（continuation）预测，且不处于调试模式下，计算并记录继续的平衡统计信息。
      if 'cont' in dists and not self.config.jax.debug_nans:
        stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
        metrics.update({f'cont_{k}': v for k, v in stats.items()})

      # 如果数据集中包含成本信息，计算并记录成本的预测和实际值的最大绝对值。
      if 'cost' in data.keys():
        metrics['cost_max_data'] = jnp.abs(data['cost']).max()
      if 'cost' in dists.keys():
        metrics['cost_max_pred'] = jnp.abs(dists['cost'].mean()).max()

      # 如果存在成本预测且数据集中包含成本信息，且不处于调试模式下，计算并记录成本的平衡统计信息。
      if 'cost' in dists and 'cost' in data.keys() and not self.config.jax.debug_nans:
        stats = jaxutils.balance_stats(dists['cost'], data['cost'], 0.1)
        metrics.update({f'cost_{k}': v for k, v in stats.items()})

      return metrics

class ImagSafeActorCritic(nj.Module):
  def __init__(self, critics, cost_critics, scales, cost_scales, act_space, config):
      # 初始化函数，用于设置策略网络的批评家、成本批评家、比例、成本比例、动作空间和配置参数
      # 过滤批评家和成本批评家，仅保留那些在scales中有对应非零比例的项
      critics = {k: v for k, v in critics.items() if scales[k]}
      cost_critics = {k: v for k, v in cost_critics.items() if scales[k]}

      # 确保每个非零比例的键在批评家和成本批评家中都存在
      for key, scale in scales.items():
        assert not scale or key in critics, key
      for key, cost_scale in cost_scales.items():
        assert not cost_scale or key in cost_critics, key

      # 初始化实例变量，仅包含那些有非零比例的批评家和成本批评家
      self.critics = {k: v for k, v in critics.items() if scales[k]}
      self.cost_critics = {k: v for k, v in cost_critics.items() if cost_scales[k]}

      # 初始化比例和成本比例以及其他配置参数
      self.scales = scales
      self.cost_scales = cost_scales
      self.act_space = act_space
      self.config = config

      # 初始化拉格朗日乘子对象，用于处理约束优化问题
      self.lagrange = jaxutils.Lagrange(self.config.lagrange_multiplier_init, self.config.penalty_multiplier_init, self.config.cost_limit, name=f'lagrange')

      # 根据动作空间的离散性选择合适的梯度计算方法
      disc = act_space.discrete
      self.grad = config.actor_grad_disc if disc else config.actor_grad_cont

      # 初始化动作网络（策略网络），根据配置参数和动作空间的形状
      self.actor = nets.MLP(
          name='actor', dims='deter', shape=act_space.shape, **config.actor,
          dist=config.actor_dist_disc if disc else config.actor_dist_cont)

      # 初始化回报和成本的归一化对象，用于稳定学习过程
      self.retnorms = {
          k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
          for k in critics}
      self.costnorms = {
          k: jaxutils.Moments(**config.costnorm, name=f'costnorm_{k}')
          for k in cost_critics}
      print("想象SafeActorCritic--优化器")
      # 初始化策略网络的优化器
      self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    for key, cost_critic in self.cost_critics.items():
      mets = cost_critic.train(traj, self.actor)
      metrics.update({f'{key}_cost_critic_{k}': v for k, v in mets.items()})
    return traj, metrics


  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    loss = loss.mean()
    # if self.config.expl_behavior not in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner'] and self.config.expl_behavior is not None:
    if self.config.task_behavior not in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner'] and self.config.expl_behavior is not None:
      print("-----------使用了SAC_Lag----------------")
      cost_advs = []
      total = sum(self.cost_scales[k] for k in self.cost_critics)
      cost_rets = []
      for key, cost_critic in self.cost_critics.items():
        cost, cost_ret, base = cost_critic.score(traj, self.actor)
        cost_rets.append(cost_ret)
        offset, invscale = self.costnorms[key](cost_ret)
        normed_ret = (cost_ret - offset) / invscale
        normed_base = (base - offset) / invscale
        cost_advs.append((normed_ret - normed_base) * self.cost_scales[key] / total)
        metrics.update(jaxutils.tensorstats(cost, f'{key}_cost'))
        metrics.update(jaxutils.tensorstats(cost_ret, f'{key}_cost_raw'))
        metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_cost_normed'))
        metrics[f'{key}_cost_rate'] = (jnp.abs(ret) >= 0.5).mean()
      if self.config.pessimistic: 
        cost_ret_episode = jnp.stack(cost_ret).sum(0)
      else:
        cost_ret_episode = jnp.stack(cost_ret).mean(0)
      penalty, lagrange_multiplier, penalty_multiplier = self.lagrange(cost_ret_episode)
      metrics[f'lagrange_multiplier'] = lagrange_multiplier
      metrics[f'penalty_multiplier'] = penalty_multiplier
      metrics[f'penalty'] = penalty
      loss += penalty
    else:
        print("-----------只使用了SAC----------------")
    return loss, metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics

class ImagActorCritic(nj.Module):
  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}

    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}

    self.scales = scales
    self.act_space = act_space
    self.config = config

    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    print("ImagActorCritic--优化器")
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
      """
      训练函数，用于更新策略和价值函数。

      参数:
      - imagine: 一个函数，用于在想象中生成轨迹。
      - start: 起始状态。
      - context: 上下文信息，未在本段代码中直接使用，可能用于其他地方。

      返回:
      - traj: 生成的轨迹。
      - metrics: 训练过程中的度量指标。
      """

      # 定义一个内部函数来计算损失
      def loss(start):
        """
        计算损失函数。

        参数:
        - start: 起始状态。

        返回:
        - loss: 计算得到的损失值。
        - (traj, metrics): 辅助返回值，包含轨迹和度量指标。
        """
        # 定义一个策略，使用actor网络生成动作
        policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
        # 使用想象函数生成轨迹
        traj = imagine(policy, start, self.config.imag_horizon)
        # 计算损失和度量指标
        loss, metrics = self.loss(traj)
        return loss, (traj, metrics)

      # 使用优化器计算损失并获取辅助返回值
      mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
      # 更新度量指标
      metrics.update(mets)

      # 对每个批评家网络进行训练
      for key, critic in self.critics.items():
        # 训练批评家网络并获取度量指标
        mets = critic.train(traj, self.actor)
        # 更新总的度量指标
        metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})

      # 返回生成的轨迹和度量指标
      return traj, metrics


  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]

class CostVFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.cost_critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None, lag=1.0):
    rew = self.rewfn(traj)
    rew_repeat = rew # * self.config.env[self.config.task.split('_')[0]].repeat
    assert len(rew_repeat) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [lag * value[-1]]
    interm = rew_repeat + lag * disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew_repeat, ret, value[:-1]