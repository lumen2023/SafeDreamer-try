import os
import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import gymnasium as gym
import traceback
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers
import datetime
import faulthandler
faulthandler.enable()


def main(argv=None):
  """
  主函数，用于处理命令行参数并根据参数执行相应的脚本。
  参数:
  - argv: 命令行参数列表，如果为None，则使用sys.argv。
  """
  from . import agent as agt

  # 解析命令行参数，分为已知参数和未知参数两部分
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  # 初始化配置，从默认配置开始
  config = embodied.Config(agt.Agent.configs['defaults'])
  # 根据解析到的配置名称更新配置
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  # 解析其他参数
  config = embodied.Flags(config).parse(other)
  # 生成当前时间字符串，用于日志目录
  now_time = datetime.datetime.now().strftime("%d-%H")
  # 构造日志目录路径，包含时间、方法、任务和种子信息
  logdir_algo = config.logdir + now_time + '_' + str(config.method)
  # 构造参数字典，包括日志目录、是否使用成本、批处理步骤数
  args = embodied.Config(
      **config.run, logdir=logdir_algo, use_cost=config.use_cost,
      batch_steps=config.batch_size * config.batch_length)
  # print(config)
  # 设置可见的CUDA设备
  os.environ['CUDA_VISIBLE_DEVICES'] = str(config.jax.logical_gpus)

  # 创建日志目录
  logdir = embodied.Path(logdir_algo)
  logdir.mkdirs()
  # 保存配置到日志目录
  config.save(logdir / 'config.yaml')
  # 创建计数器
  step = embodied.Counter()
  # 创建日志记录器
  logger = make_logger(parsed, logdir, step, config)

  # 资源清理列表，用于存储需要在finally块中关闭的对象
  cleanup = []
  # 创建PID拉格朗日控制器
  lag = PIDLagrangian(config)
  try:
    # 根据args.script的值选择执行不同的脚本
    if args.script == 'train':
      # 创建重放缓冲区、环境和代理，然后运行训练脚本
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_save':
      # 创建重放缓冲区、环境和代理，然后运行训练和保存模型的脚本
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args, lag)

    elif args.script == 'train_eval':
      # 创建重放缓冲区、评估重放缓冲区、环境和评估环境，然后运行训练和评估脚本
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      eval_env = make_envs(config)  # mode='eval'
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args, lag)

    elif args.script == 'train_holdout':
      # 创建重放缓冲区和评估重放缓冲区，根据配置选择评估重放的来源
      replay = make_replay(config, logdir / 'replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      # 创建评估环境和代理，然后运行仅评估的脚本
      env = make_envs(config, mode='eval')  # mode='eval'
      # env = make_envs(config, render_mode='human')  # mode='eval'
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args, lag)

    elif args.script == 'parallel':
      # 并行运行训练，需要确保actor_batch不超过环境数量
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      step = embodied.Counter()
      env = make_env(config)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, replay, logger, bind(make_env, config),
          num_envs=config.envs.amount, args=args)

    else:
      # 如果script值不在预定义范围内，则抛出异常
      raise NotImplementedError(args.script)
  finally:
    # 关闭所有需要清理的资源
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  """
  创建一个日志记录器。

  根据提供的参数，包括解析后的命令行参数、日志目录、步数和配置对象，
  构建一个日志记录器实例。这个日志记录器将根据配置记录到不同的输出中。

  参数:
  - parsed: 解析后的命令行参数，未使用。
  - logdir: 日志目录，所有日志数据将写入该目录。
  - step: 当前的步数，用于日志记录的步数参考。
  - config: 配置对象，包含任务和环境的配置信息。

  返回:
  一个配置好的Logger实例。
  """
  # 根据任务名称获取环境配置中的重复倍数
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)

  # 初始化Logger实例，传入步数和输出目标列表
  logger = embodied.Logger(step, [
      # 终端输出，根据配置的过滤条件记录日志到控制台
      embodied.logger.TerminalOutput(config.filter),
      # JSONL格式文件输出，记录所有指标数据到metrics.jsonl文件
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      # JSONL格式文件输出，专门记录剧集分数和成本到scores.jsonl文件
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score|episode/cost'),
      # TensorBoard输出，用于TensorBoard可视化工具
      # embodied.logger.TensorBoardOutput(logdir),
      # 以下输出方式被注释掉，表示可选的集成
      # WandB输出，用于Weights & Biases平台
      embodied.logger.WandBOutput(logdir, config),
      # MLFlow输出，用于MLflow机器学习实验管理平台
      # embodied.logger.MLFlowOutput(logdir.name),
  ], multiplier)

  # 返回配置好的Logger实例
  return logger



def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def make_envs(config, **overrides):
  suite, task = config.task.split('_', 1)
  ctors = []
  for index in range(config.envs.amount):
    ctor = lambda: make_env(config, **overrides)
    if config.envs.parallel != 'none':
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
      ctor = bind(wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
  """
  根据配置和覆盖参数创建并返回环境实例。

  参数:
    config (object): 包含任务和环境配置的对象。
    **overrides: 可选的关键字参数，用于覆盖默认的环境配置。

  返回值:
    env (object): 创建的环境实例，经过包装后返回。
  """

  # 解析任务名称，分为套件和任务两部分
  suite, task = config.task.split('_', 1)

  # 定义支持的环境类型及其对应的构造函数路径
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'New-gym': 'embodied.envs.from_gymnasium:FromGymnasium',
      'safetygym': 'embodied.envs.safetygym:SafetyGym',
      'safetygymcoor': 'embodied.envs.safetygymcoor:SafetyGymCoor',
      'safetygymmujoco': 'embodied.envs.safetygym_mujoco:SafetyGymMujoco',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
  }[suite]

  # 如果构造函数是字符串形式，则动态导入模块并获取类
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)

  # 获取环境配置，并应用覆盖参数和平台设置
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  kwargs.update({'platform': config.jax.platform})

  # 创建环境实例并进行包装
  env = ctor(task, **kwargs)
  return wrap_env(env, config)



def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env

from collections import deque

import numpy as np
# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class PIDLagrangian():  # noqa: B024
    """PID version of Lagrangian.

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    .. note::
        The PID-Lagrange is more general than the Lagrange, and can be used in any policy gradient
        algorithm. As PID_Lagrange use the PID controller to control the lagrangian multiplier, it
        is more stable than the naive Lagrange.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: `PID Lagrange <https://arxiv.org/abs/2007.03964>`_
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        config,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = config.pid.kp
        self._pid_ki: float = config.pid.ki
        self._pid_kd: float = config.pid.kd
        self._pid_d_delay = config.pid.d_delay
        self._pid_delta_p_ema_alpha: float = config.pid.delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = config.pid.delta_d_ema_alpha
        self._penalty_max: int = config.pid.penalty_max
        self._sum_norm: bool = config.pid.sum_norm
        self._diff_norm: bool = config.pid.diff_norm
        self._pid_i: float = config.pid.lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._pid_d: float = 0.0
        self._cost_limit: float = config.cost_limit
        self._cost_penalty: float = config.pid.init_penalty
        self._use_cost_decay: bool = config.pid.use_cost_decay
        self._current_cost_limit: float = config.pid.init_cost_limit
        if self._use_cost_decay:
          self._steps = [config.pid.decay_time_step * (i + 1) for i in range(config.pid.decay_num)]
          self._limits = [max(config.pid.init_cost_limit - i * config.pid.decay_limit_step,  config.cost_limit) for i in range(config.pid.decay_num)]
    @property
    def lagrange_penalty(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    @property
    def delta_p(self) -> float:
        """The lagrangian multiplier p."""
        return self._delta_p
    @property
    def pid_i(self) -> float:
        """The lagrangian multiplier i."""
        return self._pid_i

    @property
    def pid_d(self) -> float:
        """The lagrangian multiplier d."""
        return self._pid_d


    def pid_update(self, epcost, step) -> None:
        r"""Update the PID controller.

        PID controller update the lagrangian multiplier following the next equation:

        .. math::

            \lambda_{t+1} = \lambda_t + (K_p e_p + K_i \int e_p dt + K_d \frac{d e_p}{d t}) \eta

        where :math:`e_p` is the error between the current episode cost and the cost limit,
        :math:`K_p`, :math:`K_i`, :math:`K_d` are the PID parameters, and :math:`\eta` is the
        learning rate.

        Args:
            ep_cost_avg (float): The average cost of the current episode.
        """
        ep_cost_avg = epcost
        if self._use_cost_decay:
          for i, threshold in enumerate(self._steps):
            if step < threshold:
              self._current_cost_limit = self._limits[i]
              break
          else:
            self._current_cost_limit = self._cost_limit
        else:
          self._current_cost_limit = self._cost_limit

        delta = float(ep_cost_avg - self._current_cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        self._pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * self._pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
        self._cost_penalty = np.clip(self._cost_penalty, 0.0, self._penalty_max)
        return self._cost_penalty, self._pid_d, self._pid_i, self._delta_p


if __name__ == '__main__':
    try:
        # 你的主逻辑代码，例如训练、评估等
        print("程序开始运行")
        # 模拟长时间运行的任务
        main()
    except Exception as e:
        # 捕获其他所有异常并打印详细的堆栈信息
        print("程序发生了错误:")
        traceback.print_exc()  # 输出详细的堆栈信息
        print(f"错误信息: {str(e)}")
