import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import expl
from . import ninjax as nj
from . import jaxutils

import jax
from jax import lax

tree_map = jax.tree_util.tree_map

sg = lambda x: tree_map(jax.lax.stop_gradient, x)

def cost_from_state(wm, state):
  """
  根据给定的状态计算成本。

  该函数首先通过世界的模型（wm）的解码器头（'decoder'）重建状态，
  然后计算重建状态中危害（hazards）的观测值与设定的危害大小之间的距离。
  如果距离小于等于设定的危害大小，则对应的成本为1，否则为0。
  最终返回的是每个状态对应的成本。

  参数:
  - wm: 世界的模型，包含对环境的理解和预测。
  - state: 当前的状态，由环境或模型给出。

  返回:
  - cost: 每个状态对应的成本，基于状态中的危害观测计算得出。
  """
  # 通过模型的解码器头重建状态
  recon = wm.heads['decoder'](state)
  # 获取重建状态的观测值的众数
  recon = recon['observation'].mode()
  # 设定危害的大小
  hazards_size = 0.25
  # 计算批次大小
  batch_size = recon.shape[0] * recon.shape[1]
  # 提取并重塑危害的观测值
  hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
  # 计算危害的距离
  hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
    batch_size,
    -1,
  )
  # 判断危害的距离是否小于等于设定的危害大小
  condition = jnp.less_equal(hazards_dist, hazards_size)
  # 根据条件计算成本
  cost = jnp.where(condition, 1.0, 0.0)
  # 计算每个状态的总成本
  cost = cost.sum(1)
  # 可选：将成本阈值设为1，即如果成本大于等于1，则设为1，否则为0
  # condition = jnp.greater_equal(cost, 1.0)
  # cost = jnp.where(condition, 1.0, 0.0)

  # 重塑成本的形状以匹配输入状态的批次和序列长度
  cost = cost.reshape(recon.shape[0], recon.shape[1])
  # 返回计算出的成本
  return cost

class Greedy(nj.Module):

  def __init__(self, wm, act_space, config):
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    if config.use_cost:
      if config.use_cost_model:
        costfn = lambda s: wm.heads['cost'](s).mean()[1:]
      else:
        costfn = lambda s: cost_from_state(wm, s)[1:]
    if config.critic_type == 'vfunction':
      critics = {'extr': agent.VFunction(rewfn, config, name='critic')}
      if config.use_cost:
        cost_critics = {'extr': agent.CostVFunction(costfn, config, name='cost_critic')}
    else:
      raise NotImplementedError(config.critic_type)

    if config.use_cost:
      self.ac = agent.ImagSafeActorCritic(
          critics, cost_critics, {'extr': 1.0}, {'extr': 1.0}, act_space, config, name='safe_ac')
    else:
      self.ac = agent.ImagActorCritic(
          critics, {'extr': 1.0}, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    print("  ===进入贪心策略===  ")
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}


class Random(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}


class PIDPlanner(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.config = config
    self.ac = ac
    self.wm = wm
    self.act_space = act_space

    self.horizon = self.config.planner.horizon
    self.num_samples =self.config.planner.num_samples
    self.mixture_coef = self.config.planner.mixture_coef
    self.num_elites = self.config.planner.num_elites
    self.temperature = self.config.planner.temperature
    self.iterations = self.config.planner.iterations
    self.momentum = self.config.planner.momentum
    self.init_std = self.config.planner.init_std
    self.cost_limit = self.config.cost_limit
    self.cost_limit_phys = self.config.cost_limit_phys

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

  def cost_from_recon(self, recon):
      """
      从重建数据中计算成本。

      此函数的目的是评估重建数据中潜在危险的数量和大小，通过计算距离在指定大小范围内的危险来确定成本。

      参数:
      - recon: 重建数据，其形状为(batch_size, sequence_length, feature_size)。

      返回:
      - cost: 计算得到的成本，形状为(batch_size, sequence_length)。
      """
      print("PID-Planner--重建模型--cost")
      # 定义危险区域的大小阈值
      hazards_size = 0.25

      # 计算批次大小，用于后续计算
      batch_size = recon.shape[0] * recon.shape[1]

      # 提取并重塑重建数据中与危险相关的部分，以便后续处理
      hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)

      # 计算每个危险的距离，以便评估其是否在指定大小范围内
      hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
          batch_size,
          -1,
      )

      # 检查危险距离是否小于等于危险大小阈值
      condition = jnp.less_equal(hazards_dist, hazards_size)

      # 根据条件计算成本：如果危险在指定大小范围内，则成本为1，否则为0
      cost = jnp.where(condition, 1.0, 0.0)

      # 计算每个批次中至少有一个危险的成本
      cost = cost.sum(1)

      # 检查成本是否大于等于1，如果是，则将成本设置为1，否则为0
      condition = jnp.greater_equal(cost, 1.0)
      cost = jnp.where(condition, 1.0, 0.0)

      # 重塑成本数组，以匹配输入数据的批次和序列形状
      cost = cost.reshape(recon.shape[0], recon.shape[1])

      # 返回计算得到的成本
      return cost

  def true_function(self, traj_ret, traj_cost, traj_pi_gaus):
    return -traj_cost, traj_pi_gaus['action']

  def false_function(self, traj_ret, traj_cost, traj_pi_gaus):
    mask = traj_cost < self.cost_limit
    safe_elite_idxs = jnp.nonzero(lax.convert_element_type(mask, jnp.int32), size=mask.size)[0]
    return traj_ret[safe_elite_idxs], traj_pi_gaus['action'][:,safe_elite_idxs,:]

  def func_with_if_else(self, num_safe_traj, ret, cost, traj_pi_gaus):
    elite_value, elite_actions = lax.cond(num_safe_traj < self.num_elites,lambda ret, cost, traj_pi_gaus: self.true_function(ret, cost, traj_pi_gaus), lambda ret, cost, traj_pi_gaus: self.false_function(ret, cost, traj_pi_gaus), ret, cost, traj_pi_gaus)
    return elite_value, elite_actions

  def policy(self, latent, state):
    """
    Plan next action using TD-MPC inference.
    obs: raw input observation.
    eval_mode: uniform sampling and action noise is disabled during evaluation.
    step: current time step. determines e.g. planning horizon.
    t0: whether current step is the first step of an episode.
    """

    num_pi_trajs = int(self.mixture_coef * self.num_samples)
    if num_pi_trajs > 0:
      latent_pi = {k: jnp.repeat(v,num_pi_trajs,0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros(latent_pi['deter'].shape[0])
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, self.horizon)

    std = self.init_std * jnp.ones((self.horizon+1,)+self.act_space.shape)
    if 'action_mean' in state.keys():
      mean = jnp.roll(state['action_mean'], -1, axis=0)
      mean = mean.at[-1].set(mean[-2])
    else:
      mean = jnp.zeros((self.horizon+1,)+self.act_space.shape)

    for i in range(self.iterations):
      latent_gaus = {k: jnp.repeat(v,self.num_samples,0) for k, v in latent.items()}
      latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std
      def policy(s,horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        return jnp.clip(current_mean + current_std * jax.random.normal(nj.rng(),(self.num_samples,)+self.act_space.shape),-1, 1)
      traj_gaus = self.wm.imagine(policy, latent_gaus, self.horizon, use_planner=True)

      if num_pi_trajs > 0:
        traj_pi_gaus ={}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k],traj_gaus[k]],axis=1)
      else:
        traj_pi_gaus = traj_gaus
      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)

      recon = self.wm.heads['decoder'](traj_pi_gaus)

      if self.config.use_cost:
        cost, cost_ret, cost_value = self.ac.cost_critics['extr'].score(traj_pi_gaus)
        short_cost = cost.sum(0)
        traj_cost = cost.sum(0)
        # traj_cost = traj_cost * (1/ self.horizon)
        # traj_cost = traj_cost
        traj_ret_penalty = ret[0] - state['lagrange_penalty'] * cost_ret[0]
      else:
        cost = self.cost_from_recon(recon['observation'].mode())
        # [horizon,num_samples]
        # traj_cost = cost.sum(0) * (1/ self.horizon)
        traj_cost = cost.sum(0)
        # [num_samples]

      num_safe_traj = jnp.sum(lax.convert_element_type(traj_cost<self.cost_limit_phys, jnp.int32))


      elite_value, elite_actions = self.func_with_if_else(num_safe_traj, traj_ret_penalty, traj_cost, traj_pi_gaus)


      elite_idxs = jax.lax.top_k(elite_value, self.num_elites)[1]
      elite_value, elite_actions = elite_value[elite_idxs], elite_actions[:,elite_idxs,:]
      _mean = elite_actions.mean(axis=1)
      _std = elite_actions.std(axis=1)
      mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std

    a = mean[0]
    return {'action': jnp.expand_dims(a,0), 'log_action_mean': jnp.expand_dims(mean.mean(axis=0),0), 'log_action_std': jnp.expand_dims(std.mean(axis=0),0), 'log_plan_num_safe_traj': jnp.expand_dims(num_safe_traj,0), 'log_plan_ret': jnp.expand_dims(ret[0,:].mean(axis=0),0), 'log_plan_cost':jnp.expand_dims(traj_cost.mean(axis=0),0)}, {'action_mean': mean, 'action_std': std}

class CCEPlanner(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.config = config
    self.ac = ac
    self.wm = wm
    self.act_space = act_space

    self.horizon = self.config.planner.horizon
    self.num_samples =self.config.planner.num_samples
    self.mixture_coef = self.config.planner.mixture_coef
    self.mixture_coef_pi = self.config.planner.mixture_coef_pi  # pi
    self.mixture_coef_gaus = self.config.planner.mixture_coef_gaus  # gaus
    self.mixture_coef_idm = self.config.planner.mixture_coef_idm  # IDM
    self.num_elites = self.config.planner.num_elites
    self.iterations = self.config.planner.iterations
    self.momentum = self.config.planner.momentum
    self.init_std = self.config.planner.init_std
    self.cost_limit = self.config.cost_limit

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

# 维度似乎有差异
  def cost_from_recon(self, recon):
    """
    从重建数据中计算成本。

    此函数的目的是通过检查重建数据中危害区域的距离来计算成本。
    如果危害区域的距离小于或等于指定的大小，则该区域被视为不安全，会分配较高的成本。

    参数:
    - recon: 重建数据，形状为 (batch_size, sequence_length, data_dimension)。

    返回:
    - 成本矩阵，形状与输入数据的前两个维度相同。
    """
    print("CCEPlanner--重建模型--cost")
    # 定义危害区域的大小阈值
    hazards_size = 0.25

    # 计算批量大小，用于后续处理
    batch_size = recon.shape[0] * recon.shape[1]

    # 提取重建数据中危害区域的观测值，并重塑为 (batch_size, number_of_hazards, 2)
    # hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
    hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)

    # 计算危害区域观测值的距离
    hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
        batch_size,
        -1,
    )

    # 检查危害区域的距离是否小于或等于指定的大小
    condition = jnp.less_equal(hazards_dist, hazards_size)

    # 根据条件计算成本：如果距离小于或等于指定大小，则成本为1.0，否则为0.0
    cost = jnp.where(condition, 1.0, 0.0)

    # 计算每个样本的总成本
    cost = cost.sum(1)

    # 可选的进一步处理：如果成本大于等于1.0，则设置为1.0，否则为0.0
    # 这段代码被注释掉了，但保留在此以供参考
    # condition = jnp.greater_equal(cost, 1.0)
    # cost = jnp.where(condition, 1.0, 0.0)

    # 将成本重新整形为与输入数据的前两个维度相匹配的形状
    cost = cost.reshape(recon.shape[0], recon.shape[1])

    # 返回计算得到的成本
    return cost


  def true_function(self, ret, cost, traj_pi_gaus):
    """
    计算真实函数的返回值和动作序列

    参数:
        ret: 返回值参数（未在函数中使用）
        cost: 成本值，将被取反作为第一个返回值
        traj_pi_gaus: 轨迹策略高斯分布字典，包含'action'键

    返回:
        tuple: 包含两个元素的元组
            - 负的成本值
            - traj_pi_gaus字典中的动作序列
    """
    return -cost, traj_pi_gaus['action']

  def false_function(self, ret, cost, traj_pi_gaus):
    """
    根据成本限制筛选安全的精英样本并返回对应的结果和轨迹动作。

    参数:
        ret: 包含回报值的数组
        cost: 包含成本值的数组
        traj_pi_gaus: 包含轨迹动作信息的字典，其中'action'键对应动作数组

    返回:
        tuple: 包含两个元素的元组
            - 筛选后的回报值数组
            - 筛选后的轨迹动作数组
    """
    # 根据成本限制创建掩码，标记成本低于限制的样本
    mask = cost < self.config.cost_limit_phys  # 按照物理信息限制来进行约束
    # 获取满足成本限制的样本索引
    safe_elite_idxs = jnp.nonzero(lax.convert_element_type(mask, jnp.int32), size=mask.size)[0]
    # 返回筛选后的回报值和对应的动作轨迹
    return ret[safe_elite_idxs], traj_pi_gaus['action'][:,safe_elite_idxs,:]

  def func_with_if_else(self, num_safe_traj, ret, cost, traj_pi_gaus):
    """
    根据安全轨迹数量条件选择执行不同的函数来计算精英值和精英动作。

    参数:
        num_safe_traj: 安全轨迹的数量
        ret: 返回值参数
        cost: 成本参数
        traj_pi_gaus: 轨迹策略高斯分布参数

    返回:
        tuple: 包含精英值和精英动作的元组 (elite_value, elite_actions)
    """
    # 根据安全轨迹数量是否小于精英数量阈值，条件性地执行不同的函数
    elite_value, elite_actions = lax.cond(num_safe_traj < self.num_elites,
                                          lambda ret, cost, traj_pi_gaus: self.true_function(ret, cost, traj_pi_gaus),
                                          lambda ret, cost, traj_pi_gaus: self.false_function(ret, cost, traj_pi_gaus),
                                          ret, cost, traj_pi_gaus)
    return elite_value, elite_actions

  def New_policy(self, latent, state, obs = None):
    """
    使用 TD-MPC 推理规划下一步动作。

    参数:
    - latent: 包含当前潜在状态的字典。
    - state: 包含当前系统状态的字典，包括动作统计信息。

    返回值:
    - 一个包含动作及其相关统计信息的字典，以及更新后的动作均值和标准差。
    """


    ## 零、初始化参数
    idm_action = jnp.asarray(obs['action_idm'], dtype=jnp.float32).squeeze(0)  # shape (1, act_dim)

    # 定义IDM噪声水平 (可以根据需要调整，建议使用较小的值，如0.01)
    idm_noise_scale = 0.05  # 高斯噪声的标准差
    std = self.init_std * jnp.ones((self.horizon + 1,) + self.act_space.shape)

    num_pi_trajs = int(self.mixture_coef_pi * self.num_samples)
    num_gaus_trajs = int(self.mixture_coef_gaus * self.num_samples)
    num_idm_trajs = int(self.mixture_coef_idm * self.num_samples)
    num_trajs = self.num_samples
    moe_ratio_vec = jnp.array([self.mixture_coef_pi,
                               self.mixture_coef_gaus,
                               self.mixture_coef_idm], dtype=jnp.float32)  # (3,)   pi / gaus / idm  暂时使用

    if 'action_mean' in state.keys():
      # 如果存在动作均值，则滚动更新均值
      mean = jnp.roll(state['action_mean'], -1, axis=0)
      mean = mean.at[-1].set(mean[-2])
    else:
      # 否则初始化为零
      mean = jnp.zeros((self.horizon + 1,) + self.act_space.shape)

    def summarize_and_log_trajectory_statistics(traj_pi, traj_gaus, traj_idm, traj_cost, elite_idxs, config):
      import jax.numpy as jnp

      # 构造来源数组：0 = pi, 1 = gaus, 2 = idm
      src_pi = jnp.zeros(traj_pi['action'].shape[1], dtype=jnp.int32)
      src_gaus = jnp.ones(traj_gaus['action'].shape[1], dtype=jnp.int32)
      src_idm = 2 * jnp.ones(traj_idm['action'].shape[1], dtype=jnp.int32)
      src_all = jnp.concatenate([src_pi, src_gaus, src_idm], axis=0)  # (K_total,)

      # 安全轨迹掩码
      safe_mask = traj_cost < config.cost_limit_phys
      # 精英轨迹掩码（构造和 traj_cost 同维度的 0 向量，将 elite_idxs 位置设为 1）
      elite_mask = jnp.zeros_like(safe_mask).at[elite_idxs].set(1)

      def count(mask, val):
        return jnp.sum(jnp.logical_and(mask, src_all == val))

      # 统计信息 - 保持为JAX数组，不转换为Python整数
      metrics_iter = {
        'num_safe_pi': count(safe_mask, 0),
        'num_safe_gaus': count(safe_mask, 1),
        'num_safe_idm': count(safe_mask, 2),
        'num_eli_pi': count(elite_mask, 0),
        'num_eli_gaus': count(elite_mask, 1),
        'num_eli_idm': count(elite_mask, 2),
      }

      # from jax import debug
      #
      # # 构建安全轨迹统计信息字符串
      # safe_msg = "安全轨迹数量: "
      # safe_msg += "pi={num_safe_pi}、gaus={num_safe_gaus}、idm={num_safe_idm}"
      #
      # # 构建精英轨迹统计信息字符串
      # elite_msg = "精英轨迹数量: "
      # elite_msg += "pi={num_eli_pi}、gaus={num_eli_gaus}、idm={num_eli_idm}"
      #
      # # 打印安全轨迹统计信息（不换行）
      # debug.print(safe_msg,
      #             num_safe_pi=metrics_iter['num_safe_pi'],
      #             num_safe_gaus=metrics_iter['num_safe_gaus'],
      #             num_safe_idm=metrics_iter['num_safe_idm'])
      #
      # # 打印精英轨迹统计信息（添加换行符）
      # debug.print(elite_msg + "\n",
      #             num_eli_pi=metrics_iter['num_eli_pi'],
      #             num_eli_gaus=metrics_iter['num_eli_gaus'],
      #             num_eli_idm=metrics_iter['num_eli_idm'])

      return metrics_iter

    ## 一、计算从ac策略中采样的轨迹数量

    if num_pi_trajs > 0:
      # 重复潜在状态以生成多个轨迹
      latent_pi = {k: jnp.repeat(v, num_pi_trajs, 0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros(latent_pi['deter'].shape[0])

      # 定义策略函数并想象未来的轨迹
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, self.horizon) # 使用sac_lag策略生成轨迹

  ## 二、基于IDM策略的轨迹数量  把当前 latent 复制 1 次（num_idm_trajs = 1）
    latent_idm = {k: jnp.repeat(v, num_idm_trajs, 0) for k, v in latent.items()}
    latent_idm['is_terminal'] = jnp.zeros(latent_idm['deter'].shape[0])
    # 定义恒定 IDM policy：忽略 latent，仅按 horizon 返回同一动作
    def idm_policy_old(_: dict, __: int = None):
      # 返回 shape = (num_idm_trajs, act_dim)
      return jnp.repeat(idm_action[None, :], num_idm_trajs, axis=0)

    # 定义带噪声的IDM策略
    def idm_policy(_: dict, __: int = None):
      # 基础IDM动作 (shape: (num_idm_trajs, act_dim))
      base_actions = jnp.repeat(idm_action[None, :], num_idm_trajs, axis=0)

      # 生成与动作维度匹配的高斯噪声
      # 使用horizon作为随机数生成的子键，确保每个时间步的噪声不同
      noise_key = nj.rng()  # 获取随机数生成器
      noise = idm_noise_scale * jax.random.normal(noise_key, shape=base_actions.shape)

      # 应用噪声并裁剪到动作空间范围
      noisy_actions = jnp.clip(base_actions + noise, -1.0, 1.0)

      return noisy_actions

    # 生成完整 IDM 轨迹（T+1, 1, …）
    traj_idm = self.wm.imagine(idm_policy, latent_idm, self.horizon)

    # 三、迭代 self.iterations 次的 CCE 主循环
    for i in range(self.iterations):
      # 重复潜在状态以生成高斯分布的轨迹
      latent_gaus = {k: jnp.repeat(v, num_gaus_trajs, 0) for k, v in latent.items()}
      latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std

      # 定义高斯分布的策略函数
      def policy(s, horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        return jnp.clip(
          current_mean + current_std * jax.random.normal(nj.rng(), (num_gaus_trajs,) + self.act_space.shape), -1, 1)

      # 想象未来的轨迹并使用规划器
      traj_gaus = self.wm.imagine(policy, latent_gaus, self.horizon, use_planner=True)

      if num_pi_trajs > 0:
        # 如果有策略轨迹，将策略轨迹和高斯轨迹合并
        traj_pi_gaus = {}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k], traj_gaus[k], traj_idm[k]], axis=1)
      else:
        traj_pi_gaus = traj_gaus

      # 计算奖励、回报和价值
      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)
      traj_rew = ret[0]

      if self.config.use_cost:
        # 如果使用成本计算，评估成本并调整成本
        cost, cost_ret, cost_value = self.ac.cost_critics['extr'].score(traj_pi_gaus)
        # traj_cost = cost.sum(0) * (2 / self.horizon)
        traj_cost = cost.sum(0)

      else:
        # 否则从重建中计算成本
        recon = self.wm.heads['decoder'](traj_pi_gaus)
        # print("Debug: recon keys ->", recon.keys())  # 打印 recon 的键
        cost = self.cost_from_recon(recon['observation'].mode())
        # cost = self.cost_from_recon(recon['image'].mode())
        traj_cost = cost.sum(0)

      # 计算安全轨迹的数量
      num_safe_traj = jnp.sum(lax.convert_element_type(traj_cost < self.config.cost_limit_phys, jnp.int32))

      # 根据安全轨迹选择精英动作
      elite_value, elite_actions = self.func_with_if_else(num_safe_traj, traj_rew, traj_cost, traj_pi_gaus)

      # 获取精英动作的索引并更新均值和标准差
      elite_idxs = jax.lax.top_k(elite_value, self.num_elites)[1]
      elite_value, elite_actions = elite_value[elite_idxs], elite_actions[:, elite_idxs, :]
      _mean = elite_actions.mean(axis=1)
      _std = elite_actions.std(axis=1)
      mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std

    metrics_iter = summarize_and_log_trajectory_statistics(
      traj_pi=traj_pi, traj_gaus=traj_gaus, traj_idm=traj_idm,
      traj_cost=traj_cost, elite_idxs=elite_idxs, config=self.config
    )

    # 获取最终的动作并返回结果
    a = mean[0]
    print("进入CCEplanner策略当中")
    return {
      'action': jnp.expand_dims(a, 0),
      'log_action_mean': jnp.expand_dims(mean.mean(axis=0), 0),
      'log_action_std': jnp.expand_dims(std.mean(axis=0), 0),
      'log_plan_num_safe_traj': jnp.expand_dims(num_safe_traj, 0),
      'log_plan_ret': jnp.expand_dims(ret[0, :].mean(axis=0), 0),
      'log_plan_cost': jnp.expand_dims(traj_cost.mean(axis=0), 0),

      # —— 新增：MoE 比例（step 级）——
      'log_moe_ratio': jnp.expand_dims(moe_ratio_vec, 0),  # shape (1, 3)

      # —— 新增：各策略轨迹条数（step 级）——
      'log_plan_num_traj': jnp.expand_dims(jnp.int32(num_trajs), 0),  # (1,)
      'log_plan_num_pi': jnp.expand_dims(jnp.int32(num_pi_trajs), 0),  # (1,)
      'log_plan_num_gaus': jnp.expand_dims(jnp.int32(num_gaus_trajs), 0),  # (1,)
      'log_plan_num_idm': jnp.expand_dims(jnp.int32(num_idm_trajs), 0),  # (1,)

      # 新增的统计指标
      'log_plan_num_safe_pi': jnp.expand_dims(metrics_iter['num_safe_pi'], 0),
      'log_plan_num_safe_gaus': jnp.expand_dims(metrics_iter['num_safe_gaus'], 0),
      'log_plan_num_safe_idm': jnp.expand_dims(metrics_iter['num_safe_idm'], 0),
      'log_plan_num_eli_pi': jnp.expand_dims(metrics_iter['num_eli_pi'], 0),
      'log_plan_num_eli_gaus': jnp.expand_dims(metrics_iter['num_eli_gaus'], 0),
      'log_plan_num_eli_idm': jnp.expand_dims(metrics_iter['num_eli_idm'], 0)

    }, {'action_mean': mean, 'action_std': std}

  def policy(self, latent, state, obs = None):
    """
    使用 TD-MPC 推理规划下一步动作。

    参数:
    - latent: 包含当前潜在状态的字典。
    - state: 包含当前系统状态的字典，包括动作统计信息。

    返回值:
    - 一个包含动作及其相关统计信息的字典，以及更新后的动作均值和标准差。
    """


    ## 零、初始化参数
    idm_action = jnp.asarray(obs['action_idm'], dtype=jnp.float32).squeeze(0)  # shape (1, act_dim)

    # 定义IDM噪声水平 (可以根据需要调整，建议使用较小的值，如0.01)
    idm_noise_scale = 0.05  # 高斯噪声的标准差
    std = self.init_std * jnp.ones((self.horizon + 1,) + self.act_space.shape)

    num_pi_trajs = int(self.mixture_coef_pi * self.num_samples)
    num_gaus_trajs = int(self.mixture_coef_gaus * self.num_samples)
    num_idm_trajs = int(self.mixture_coef_idm * self.num_samples)
    num_trajs = self.num_samples
    moe_ratio_vec = jnp.array([self.mixture_coef_pi,
                               self.mixture_coef_gaus,
                               self.mixture_coef_idm], dtype=jnp.float32)  # (3,)   pi / gaus / idm  暂时使用

    if 'action_mean' in state.keys():
      # 如果存在动作均值，则滚动更新均值
      mean = jnp.roll(state['action_mean'], -1, axis=0)
      mean = mean.at[-1].set(mean[-2])
    else:
      # 否则初始化为零
      mean = jnp.zeros((self.horizon + 1,) + self.act_space.shape)

    def summarize_and_log_trajectory_statistics(traj_pi, traj_gaus, traj_idm, traj_cost, elite_idxs, config):
      import jax.numpy as jnp

      # 构造来源数组：0 = pi, 1 = gaus, 2 = idm
      src_pi = jnp.zeros(traj_pi['action'].shape[1], dtype=jnp.int32)
      src_gaus = jnp.ones(traj_gaus['action'].shape[1], dtype=jnp.int32)
      src_idm = 2 * jnp.ones(traj_idm['action'].shape[1], dtype=jnp.int32)
      src_all = jnp.concatenate([src_pi, src_gaus, src_idm], axis=0)  # (K_total,)

      # 安全轨迹掩码
      safe_mask = traj_cost < config.cost_limit_phys
      # 精英轨迹掩码（构造和 traj_cost 同维度的 0 向量，将 elite_idxs 位置设为 1）
      elite_mask = jnp.zeros_like(safe_mask).at[elite_idxs].set(1)

      def count(mask, val):
        return jnp.sum(jnp.logical_and(mask, src_all == val))

      # 统计信息 - 保持为JAX数组，不转换为Python整数
      metrics_iter = {
        'num_safe_pi': count(safe_mask, 0),
        'num_safe_gaus': count(safe_mask, 1),
        'num_safe_idm': count(safe_mask, 2),
        'num_eli_pi': count(elite_mask, 0),
        'num_eli_gaus': count(elite_mask, 1),
        'num_eli_idm': count(elite_mask, 2),
      }

      # from jax import debug
      #
      # # 构建安全轨迹统计信息字符串
      # safe_msg = "安全轨迹数量: "
      # safe_msg += "pi={num_safe_pi}、gaus={num_safe_gaus}、idm={num_safe_idm}"
      #
      # # 构建精英轨迹统计信息字符串
      # elite_msg = "精英轨迹数量: "
      # elite_msg += "pi={num_eli_pi}、gaus={num_eli_gaus}、idm={num_eli_idm}"
      #
      # # 打印安全轨迹统计信息（不换行）
      # debug.print(safe_msg,
      #             num_safe_pi=metrics_iter['num_safe_pi'],
      #             num_safe_gaus=metrics_iter['num_safe_gaus'],
      #             num_safe_idm=metrics_iter['num_safe_idm'])
      #
      # # 打印精英轨迹统计信息（添加换行符）
      # debug.print(elite_msg + "\n",
      #             num_eli_pi=metrics_iter['num_eli_pi'],
      #             num_eli_gaus=metrics_iter['num_eli_gaus'],
      #             num_eli_idm=metrics_iter['num_eli_idm'])

      return metrics_iter

    ## 一、计算从ac策略中采样的轨迹数量

    if num_pi_trajs > 0:
      # 重复潜在状态以生成多个轨迹
      latent_pi = {k: jnp.repeat(v, num_pi_trajs, 0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros(latent_pi['deter'].shape[0])

      # 定义策略函数并想象未来的轨迹
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, self.horizon) # 使用sac_lag策略生成轨迹

  ## 二、基于IDM策略的轨迹数量  把当前 latent 复制 1 次（num_idm_trajs = 1）
    latent_idm = {k: jnp.repeat(v, num_idm_trajs, 0) for k, v in latent.items()}
    latent_idm['is_terminal'] = jnp.zeros(latent_idm['deter'].shape[0])
    # 定义恒定 IDM policy：忽略 latent，仅按 horizon 返回同一动作
    def idm_policy_old(_: dict, __: int = None):
      # 返回 shape = (num_idm_trajs, act_dim)
      return jnp.repeat(idm_action[None, :], num_idm_trajs, axis=0)

    # 定义带噪声的IDM策略
    def idm_policy(_: dict, __: int = None):
      # 基础IDM动作 (shape: (num_idm_trajs, act_dim))
      base_actions = jnp.repeat(idm_action[None, :], num_idm_trajs, axis=0)

      # 生成与动作维度匹配的高斯噪声
      # 使用horizon作为随机数生成的子键，确保每个时间步的噪声不同
      noise_key = nj.rng()  # 获取随机数生成器
      noise = idm_noise_scale * jax.random.normal(noise_key, shape=base_actions.shape)

      # 应用噪声并裁剪到动作空间范围
      noisy_actions = jnp.clip(base_actions + noise, -1.0, 1.0)

      return noisy_actions

    # 生成完整 IDM 轨迹（T+1, 1, …）
    traj_idm = self.wm.imagine(idm_policy, latent_idm, self.horizon)

    # 三、迭代 self.iterations 次的 CCE 主循环
    for i in range(self.iterations):
      # 重复潜在状态以生成高斯分布的轨迹
      latent_gaus = {k: jnp.repeat(v, num_gaus_trajs, 0) for k, v in latent.items()}
      latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std

      # 定义高斯分布的策略函数
      def policy(s, horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        return jnp.clip(
          current_mean + current_std * jax.random.normal(nj.rng(), (num_gaus_trajs,) + self.act_space.shape), -1, 1)

      # 想象未来的轨迹并使用规划器
      traj_gaus = self.wm.imagine(policy, latent_gaus, self.horizon, use_planner=True)

      if num_pi_trajs > 0:
        # 如果有策略轨迹，将策略轨迹和高斯轨迹合并
        traj_pi_gaus = {}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k], traj_gaus[k], traj_idm[k]], axis=1)
      else:
        traj_pi_gaus = traj_gaus

      # 计算奖励、回报和价值
      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)
      traj_rew = ret[0]

      if self.config.use_cost:
        # 如果使用成本计算，评估成本并调整成本
        cost, cost_ret, cost_value = self.ac.cost_critics['extr'].score(traj_pi_gaus)
        # traj_cost = cost.sum(0) * (2 / self.horizon)
        traj_cost = cost.sum(0)

      else:
        # 否则从重建中计算成本
        recon = self.wm.heads['decoder'](traj_pi_gaus)
        # print("Debug: recon keys ->", recon.keys())  # 打印 recon 的键
        cost = self.cost_from_recon(recon['observation'].mode())
        # cost = self.cost_from_recon(recon['image'].mode())
        traj_cost = cost.sum(0)

      # 计算安全轨迹的数量
      num_safe_traj = jnp.sum(lax.convert_element_type(traj_cost < self.config.cost_limit_phys, jnp.int32))

      # 根据安全轨迹选择精英动作
      elite_value, elite_actions = self.func_with_if_else(num_safe_traj, traj_rew, traj_cost, traj_pi_gaus)

      # 获取精英动作的索引并更新均值和标准差
      elite_idxs = jax.lax.top_k(elite_value, self.num_elites)[1]
      elite_value, elite_actions = elite_value[elite_idxs], elite_actions[:, elite_idxs, :]
      _mean = elite_actions.mean(axis=1)
      _std = elite_actions.std(axis=1)
      mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std

    metrics_iter = summarize_and_log_trajectory_statistics(
      traj_pi=traj_pi, traj_gaus=traj_gaus, traj_idm=traj_idm,
      traj_cost=traj_cost, elite_idxs=elite_idxs, config=self.config
    )

    # 获取最终的动作并返回结果
    a = mean[0]
    print("进入CCEplanner策略当中")
    return {
      'action': jnp.expand_dims(a, 0),
      'log_action_mean': jnp.expand_dims(mean.mean(axis=0), 0),
      'log_action_std': jnp.expand_dims(std.mean(axis=0), 0),
      'log_plan_num_safe_traj': jnp.expand_dims(num_safe_traj, 0),
      'log_plan_ret': jnp.expand_dims(ret[0, :].mean(axis=0), 0),
      'log_plan_cost': jnp.expand_dims(traj_cost.mean(axis=0), 0),

      # —— 新增：MoE 比例（step 级）——
      'log_moe_ratio': jnp.expand_dims(moe_ratio_vec, 0),  # shape (1, 3)

      # —— 新增：各策略轨迹条数（step 级）——
      'log_plan_num_traj': jnp.expand_dims(jnp.int32(num_trajs), 0),  # (1,)
      'log_plan_num_pi': jnp.expand_dims(jnp.int32(num_pi_trajs), 0),  # (1,)
      'log_plan_num_gaus': jnp.expand_dims(jnp.int32(num_gaus_trajs), 0),  # (1,)
      'log_plan_num_idm': jnp.expand_dims(jnp.int32(num_idm_trajs), 0),  # (1,)

      # 新增的统计指标
      'log_plan_num_safe_pi': jnp.expand_dims(metrics_iter['num_safe_pi'], 0),
      'log_plan_num_safe_gaus': jnp.expand_dims(metrics_iter['num_safe_gaus'], 0),
      'log_plan_num_safe_idm': jnp.expand_dims(metrics_iter['num_safe_idm'], 0),
      'log_plan_num_eli_pi': jnp.expand_dims(metrics_iter['num_eli_pi'], 0),
      'log_plan_num_eli_gaus': jnp.expand_dims(metrics_iter['num_eli_gaus'], 0),
      'log_plan_num_eli_idm': jnp.expand_dims(metrics_iter['num_eli_idm'], 0)

    }, {'action_mean': mean, 'action_std': std}


class CEMPlanner(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.ac = ac
    self.wm = wm
    self.act_space = act_space
    self.config = config
  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

  def policy(self, latent, state):
    """
    使用TD-MPC推理规划下一步动作。

    参数:
    - latent: 包含当前潜在状态的字典。
    - state: 包含当前系统状态的字典，包括动作统计信息。

    返回值:
    - 一个包含动作及其相关统计信息的字典，以及更新后的动作均值和标准差。
    """

    # 加载配置参数
    horizon = self.config.planner.horizon  # 规划时间范围
    num_samples = self.config.planner.num_samples  # 样本数量
    mixture_coef = self.config.planner.mixture_coef  # 混合系数
    num_elites = self.config.planner.num_elites  # 精英样本数量
    temperature = self.config.planner.temperature  # 温度参数
    iterations = self.config.planner.iterations  # 迭代次数
    momentum = self.config.planner.momentum  # 动量参数
    init_std = self.config.planner.init_std  # 初始化标准差
    cost_limit = self.config.cost_limit  # 成本限制

    # 计算从策略中采样的轨迹数量
    num_pi_trajs = int(mixture_coef * num_samples)
    if num_pi_trajs > 0:
      # 重复潜在状态以生成多个轨迹
      latent_pi = {k: jnp.repeat(v, num_pi_trajs, 0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros(latent_pi['deter'].shape[0])

      # 定义策略函数并想象未来的轨迹
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, horizon)

    # 初始化动作的标准差
    std = init_std * jnp.ones((horizon + 1,) + self.act_space.shape)

    if 'action_mean' in state.keys():
      # 如果存在动作均值，则滚动更新均值
      mean = jnp.roll(state['action_mean'], -1, axis=0)
      mean = mean.at[-1].set(mean[-2])
    else:
      # 否则初始化为零
      mean = jnp.zeros((horizon + 1,) + self.act_space.shape)

    for i in range(iterations):
      # 重复潜在状态以生成高斯分布的轨迹
      latent_gaus = {k: jnp.repeat(v, num_samples, 0) for k, v in latent.items()}
      latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std

      # 定义高斯分布的策略函数
      def policy(s, horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        action = jnp.clip(
          current_mean + current_std * jax.random.normal(nj.rng(), (num_samples,) + self.act_space.shape), -1, 1)
        return action

      # 想象未来的轨迹并使用规划器
      traj_gaus = self.wm.imagine(policy, latent_gaus, horizon, use_planner=True)

      if num_pi_trajs > 0:
        # 如果有策略轨迹，将策略轨迹和高斯轨迹合并
        traj_pi_gaus = {}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k], traj_gaus[k]], axis=1)
      else:
        traj_pi_gaus = traj_gaus

      # 计算奖励、回报和价值
      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)

      # 计算轨迹奖励总和
      traj_rew = rew.sum(0)

      # 获取精英索引及对应的奖励和动作
      elite_idxs = jax.lax.top_k(traj_rew, num_elites)[1]
      elite_value, elite_actions = traj_rew[elite_idxs], traj_pi_gaus['action'][:, elite_idxs, :]

      # 更新参数
      max_value = elite_value.max()
      score = jnp.exp(temperature * (elite_value - max_value))
      score /= score.sum(0)
      _mean = jnp.sum(jnp.expand_dims(score, 1) * elite_actions, axis=1) / (score.sum(0) + 1e-9)
      _std = jnp.sqrt(jnp.sum(jnp.expand_dims(score, 1) * (elite_actions - jnp.expand_dims(_mean, 1)) ** 2, axis=1) / (
                score.sum(0) + 1e-9))
      mean, std = momentum * mean + (1 - momentum) * _mean, _std

    # 获取最终的动作并返回结果
    a = mean[0]
    return {
      'action': jnp.expand_dims(a, 0),
      'log_action_mean': jnp.expand_dims(mean.mean(axis=0), 0),
      'log_action_std': jnp.expand_dims(std.mean(axis=0), 0),
      'log_plan_num_safe_traj': jnp.zeros(1),
      'log_plan_ret': jnp.expand_dims(ret[0, :].mean(axis=0), 0),
      'log_plan_cost': jnp.zeros(1)
    }, {'action_mean': mean, 'action_std': std}


class CEMPlanner_parallel(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.ac = ac
    self.wm = wm
    self.act_space = act_space
    self.config = config
  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

  def policy(self, latent, state):
    """
    Plan next action using TD-MPC inference.
    obs: raw input observation.
    eval_mode: uniform sampling and action noise is disabled during evaluation.
    step: current time step. determines e.g. planning horizon.
    t0: whether current step is the first step of an episode.
    """
    horizon = self.config.planner.horizon
    num_samples =self.config.planner.num_samples
    mixture_coef = self.config.planner.mixture_coef
    num_elites = self.config.planner.num_elites
    temperature = self.config.planner.temperature
    iterations = self.config.planner.iterations
    momentum = self.config.planner.momentum
    init_std = self.config.planner.init_std
    cost_limit = self.config.cost_limit

    num_pi_trajs = int(mixture_coef * num_samples)
    env_amount = latent['deter'].shape[0]
    latent = {k: jnp.expand_dims(v,0) for k, v in latent.items()}
    if num_pi_trajs > 0:
      latent_pi = {k: jnp.repeat(v, num_pi_trajs,0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros((latent_pi['deter'].shape[0], latent_pi['deter'].shape[1]))
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, horizon)

    std = init_std * jnp.ones((env_amount, horizon+1,)+self.act_space.shape)
    if 'action_mean' in state.keys():
      mean = jnp.roll(state['action_mean'], -1, axis=1)
      mean = mean.at[:, -1].set(mean[:, -2])
    else:
      mean = jnp.zeros((env_amount, horizon+1,)+self.act_space.shape)


    for i in range(iterations):
      latent_gaus = {k: jnp.repeat(v,num_samples,0) for k, v in latent.items()} 
      latent_gaus['is_terminal'] = jnp.zeros((latent_gaus['deter'].shape[0], latent_gaus['deter'].shape[1]))
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std
      def policy(s,horizon):
        current_mean = s['action_mean'][:, horizon]
        current_std = s['action_std'][:, horizon]
        noise = jax.random.normal(nj.rng(),(num_samples, env_amount,) + self.act_space.shape)
        action = jnp.clip(jnp.expand_dims(current_mean,0)+ jnp.expand_dims(current_std,0) * noise, -1, 1)
        return action
      traj_gaus = self.wm.imagine(policy, latent_gaus, horizon, use_planner=True)

      if num_pi_trajs > 0:
        traj_pi_gaus ={}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k],traj_gaus[k]],axis=1)


      elif num_pi_trajs == 0:
        traj_pi_gaus = traj_gaus
      else:
        raise NotImplementedError

      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)

      ret_weight = ret * sg(traj_pi_gaus['weight'])[:-1]



      sum_rew = rew.sum(0)

      traj_reward = jnp.transpose(ret_weight[0,:,:]) # [num_samples, env.amount] to [env.amount, num_samples]
      elite_idxs = jax.lax.top_k(traj_reward, num_elites)[1] # [env.amount, k]
      elite_value = jnp.zeros((env_amount, num_elites, 1))
      elite_actions = jnp.zeros((env_amount, horizon+1, num_elites, self.act_space.shape[0]))
      for i in range(env_amount):
        elite_value = elite_value.at[i].set(jnp.expand_dims(traj_reward[i, elite_idxs[i]],axis=1))
        env_actions = traj_pi_gaus['action'][:, :, i, :] # [horizon+1, num_gaus+num_pim, action_dim]
        env_actions = env_actions[:, elite_idxs[i], :]
        elite_actions = elite_actions.at[i].set(env_actions) #[amount, k]
      # Update parameters
      max_value = elite_value.max(axis=1)
      score = jnp.exp(temperature*(elite_value - jnp.expand_dims(max_value,axis=1)))
      score /= jnp.expand_dims(score.sum(axis=1), axis=1) # [amount, k], elite_actions: [horizon,amount,k,dim]
      _mean = jnp.sum(jnp.expand_dims(score, axis=1) * elite_actions , axis=2) / jnp.expand_dims((score.sum(axis=1) + 1e-9), axis=1) # [horizon,amount,dim]
      _std = jnp.sqrt(jnp.sum(jnp.expand_dims(score, axis=1) * (elite_actions - jnp.expand_dims(_mean,2)) ** 2, axis=2) / jnp.expand_dims(score.sum(axis=1) + 1e-9, axis=1))

      mean, std = momentum * mean + (1 - momentum) * _mean, _std
      # (env_amount, num_elites, 1)

    a = mean[:,0,:]
    return {'action': a, 'log_plan_action_mean': mean.mean(axis=1), 'log_plan_action_std': std.mean(axis=1), 'log_plan_num_safe_traj': jnp.zeros(env_amount), 'log_plan_ret': traj_reward[:,:].mean(axis=1), 'log_plan_cost': jnp.zeros(env_amount)}, {'action_mean': mean, 'action_std': std} #{'action_mean': mean, 'action_std': std, 'plan_num_safe_traj': jnp.zeros(env_amount), 'plan_cost':jnp.zeros(env_amount), 'plan_ret': traj_reward[:,:].mean(axis=1)}
