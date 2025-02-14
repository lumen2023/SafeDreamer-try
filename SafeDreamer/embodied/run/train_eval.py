import re

import embodied
import numpy as np

class CostEma:

  def __init__(self, initial=0):
    self.value = initial

class Arrive:

  def __init__(self):
    self.value = []

def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args, lag):
  """训练和评估一个智能体。

  参数:
  - agent: 智能体对象，负责执行策略和学习。
  - train_env: 训练环境对象。
  - eval_env: 评估环境对象。
  - train_replay: 训练回放缓冲区。
  - eval_replay: 评估回放缓冲区。
  - logger: 日志记录器对象，用于记录训练和评估的指标。
  - args: 包含配置参数的对象。
  - lag: PID控制器对象，用于调整训练过程中的成本。

  返回值:
  无
  """

  # 初始化日志目录并创建目录
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)

  # 定义各种条件判断函数
  should_expl = embodied.when.Until(args.expl_until)  # 探索阶段结束时间
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)  # 是否进行训练的条件
  should_log = embodied.when.Clock(args.log_every)  # 是否记录日志的条件
  should_save = embodied.when.Clock(args.save_every)  # 是否保存检查点的条件
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)  # 是否进行评估的条件
  should_sync = embodied.when.Every(args.sync_every)  # 是否同步模型的条件

  # 初始化计数器和其他辅助变量
  step = logger.step
  cost_ema = CostEma(0.0)  # 成本指数移动平均
  train_arrive_num = Arrive()  # 记录训练环境中到达目的地的情况
  eval_arrive_num = Arrive()  # 记录评估环境中到达目的地的情况
  updates = embodied.Counter()  # 更新次数计数器
  metrics = embodied.Metrics()  # 记录训练和评估的度量

  # 打印环境的观测空间和动作空间
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  # 初始化计时器并包装需要计时的方法
  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()

  def per_episode(ep, mode):
    """处理每个episode的数据并记录相关指标。

    参数:
    - ep: episode数据字典。
    - mode: 模式（'train' 或 'eval'）。
    """
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')

    if 'cost' in ep.keys():
      cost = float(ep['cost'].astype(np.float64).sum())
      logger.add({
          'cost': cost,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      print(f'Episode has {length} steps and cost {cost:.1f}.')
      cost_ema.value = cost_ema.value * 0.99 + cost * 0.01
      logger.add({
          'cost_ema': cost_ema.value,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      if step > 5000:
        lag.pid_update(cost_ema.value, step)

    if 'arrive_dest' in ep.keys():
      if mode == 'train':
        train_arrive_num.value.append(int(ep['arrive_dest'][-1]))
        if len(train_arrive_num.value) == 10:
          arrive_rate = sum(train_arrive_num.value) / 10
          train_arrive_num.value = []
          logger.add({
              'arrive_rate': arrive_rate,
          }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
          print(f'train 10 episodes has average arrive rate {arrive_rate:.2f}.')

      else:
        eval_arrive_num.value.append(int(ep['arrive_dest'][-1]))
        if len(eval_arrive_num.value) == 10:
          arrive_rate = sum(eval_arrive_num.value) / 10
          eval_arrive_num.value = []
          logger.add({
              'arrive_rate': arrive_rate,
          }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
          print(f'eval 10 episodes has average arrive rate {arrive_rate:.2f}.')

    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=f'{mode}_stats')

  # 创建驱动程序以与环境交互
  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  # 使用随机策略填充初始回放缓冲区
  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
  logger.add(metrics.result())
  logger.write()

  # 准备训练和评估的数据集
  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # 用于在训练步骤中可写的状态
  batch = [None]

  def train_step(tran, worker):
    """执行训练步骤。

    参数:
    - tran: 过渡数据。
    - worker: 工作者标识符。
    """
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)

  driver_train.on_step(train_step)

  # 设置检查点以保存和加载模型
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # 注册已保存检查点

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')

  # 开始主训练循环
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps), lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
    driver_train(policy_train, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()


