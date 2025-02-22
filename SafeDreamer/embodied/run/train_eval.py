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

  该函数在一个训练环境中训练智能体，并在单独的评估环境中对其进行评估。
  它定期记录指标并保存智能体。

  Args:
    agent: 智能体对象，用于与环境交互。
    train_env: 训练环境对象。
    eval_env: 评估环境对象。
    train_replay: 训练回放缓冲区。
    eval_replay: 评估回放缓冲区。
    logger: 日志记录器对象，用于记录训练和评估过程中的各种指标。
    args: 参数对象，包含配置参数。
    lag: PID控制器对象，用于处理成本约束。

  Returns:
    无返回值。该函数通过日志记录器记录训练和评估过程中的各种指标。
  """

  # 初始化日志目录并创建目录
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)

  # 初始化条件判断器
  should_expl = embodied.when.Until(args.expl_until)  # 探索阶段
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)  # 训练频率
  should_log = embodied.when.Clock(args.log_every)  # 日志记录频率
  should_save = embodied.when.Clock(args.save_every)  # 模型保存频率
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)  # 评估频率
  should_sync = embodied.when.Every(args.sync_every)  # 同步频率

  # 初始化计数器和其他辅助变量
  step = logger.step
  cost_ema = CostEma(0.0)  # 成本指数移动平均
  train_arrive_num = Arrive()  # 训练到达次数
  eval_arrive_num = Arrive()  # 评估到达次数
  updates = embodied.Counter()  # 更新计数器
  metrics = embodied.Metrics()  # 指标收集器

  # 打印环境信息
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
    """处理每个episode结束后的数据记录。

    Args:
      ep: episode数据字典。
      mode: 模式（'train' 或 'eval'）。
    """
    # 计算episode的长度和得分
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    speed_averge = float(ep['speed'].astype(np.float64).sum()) / length
    # 将长度和得分记录到日志中
    logger.add({
        'length': length,
        'score': score,
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    # if ep['crash'][-1] == False:
    #   arrive_dest = True
    # else:
    #   arrive_dest = False
    # 判断是否到达目的地
    arrive_dest = not ep['crash'][-1]
    # 将长度和平均速度记录到日志中
    logger.add({
        'speed_averge': speed_averge,
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and 平均速度为： {speed_averge:.1f}.')

    # 如果episode数据中包含成本信息，则计算并记录成本
    if 'cost' in ep.keys():
      cost = float(ep['cost'].astype(np.float64).sum())
      logger.add({
          'cost': cost,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      print(f'Episode has {length} steps and cost {cost:.1f}.')
      # 更新成本的指数移动平均值
      cost_ema.value = cost_ema.value * 0.99 + cost * 0.01
      logger.add({
          'cost_ema': cost_ema.value,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      if step > 5000:
        lag.pid_update(cost_ema.value, step)

    # 如果episode数据中包含是否到达目的地的信息，则计算并记录到达率
    # 检查episode数据中是否包含到达目的地的信息
    # if 'arrive_dest' in ep.keys():
    #   # 打印提示信息，表示有到达目的地的事件发生
    #   print('----!!!!!!!!!!!!!!!!!!arrive_dest!!!!!!!!!!!!!!!----')
    #   # 根据模式的不同（训练或评估），处理到达目的地的数据
    #   if mode == 'train':
    #     # 在训练模式下，将到达目的地的信息添加到训练到达数量列表中
    #     train_arrive_num.value.append(int(ep['arrive_dest'][-1]))
    #     # 当训练到达数量列表中的数据达到10条时，计算平均到达率并重置列表
    #     if len(train_arrive_num.value) == 10:
    #       arrive_rate = sum(train_arrive_num.value) / 10
    #       train_arrive_num.value = []
    #       # 将计算得到的平均到达率记录到日志中
    #       logger.add({
    #           'arrive_rate': arrive_rate,
    #       }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    #       # 打印训练模式下10个episode的平均到达率
    #       print(f'train 10 episodes has average arrive rate {arrive_rate:.2f}.')
    #   else:
    #     # 在评估模式下，将到达目的地的信息添加到评估到达数量列表中
    #     eval_arrive_num.value.append(int(ep['arrive_dest'][-1]))
    #     # 当评估到达数量列表中的数据达到10条时，计算平均到达率并重置列表
    #     if len(eval_arrive_num.value) == 10:
    #       arrive_rate = sum(eval_arrive_num.value) / 10
    #       eval_arrive_num.value = []
    #       # 将计算得到的平均到达率记录到日志中
    #       logger.add({
    #           'arrive_rate': arrive_rate,
    #       }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    #       # 打印评估模式下10个episode的平均到达率
    #       print(f'eval 10 episodes has average arrive rate {arrive_rate:.2f}.')

    # 根据模式的不同（训练或评估），处理到达目的地的数据
    # print('----!!!!!!!!!!!!!!!!!!arrive_dest!!!!!!!!!!!!!!!----')
    if mode == 'train':
      # 在训练模式下，将到达目的地的信息添加到训练到达数量列表中
      train_arrive_num.value.append(int(arrive_dest))
      # 当训练到达数量列表中的数据达到10条时，计算平均到达率并重置列表
      if len(train_arrive_num.value) == 10:
        arrive_rate = sum(train_arrive_num.value) / 10
        train_arrive_num.value = []
        # 将计算得到的平均到达率记录到日志中
        logger.add({
          'arrive_rate': arrive_rate,
        }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
        # 打印训练模式下10个episode的平均到达率
        print(f'train 10 episodes has average arrive rate {arrive_rate:.2f}.')
    else:
      # 在评估模式下，将到达目的地的信息添加到评估到达数量列表中
      eval_arrive_num.value.append(int(arrive_dest))
      # 当评估到达数量列表中的数据达到10条时，计算平均到达率并重置列表
      if len(eval_arrive_num.value) == 10:
        arrive_rate = sum(eval_arrive_num.value) / 10
        eval_arrive_num.value = []
        # 将计算得到的平均到达率记录到日志中
        logger.add({
          'arrive_rate': arrive_rate,
        }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
        # 打印评估模式下10个episode的平均到达率
        print(f'eval 10 episodes has average arrive rate {arrive_rate:.2f}.')

    # 收集并记录episode中的其他统计信息
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

  # 创建训练环境的驱动器（Driver），用于与环境交互并收集数据。
  driver_train = embodied.Driver(train_env)
  # 在每个episode结束时调用回调函数，记录训练模式下的episode信息。
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  # 在每一步骤后调用回调函数，增加步数计数器。
  driver_train.on_step(lambda tran, _: step.increment())
  # 在每一步骤后将数据添加到训练回放缓冲区。
  driver_train.on_step(train_replay.add)

  # 创建评估环境的驱动器（Driver），用于与环境交互并收集数据。
  driver_eval = embodied.Driver(eval_env)
  # 在每一步骤后将数据添加到评估回放缓冲区。
  driver_eval.on_step(eval_replay.add)
  # 在每个episode结束时调用回调函数，记录评估模式下的episode信息。
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  # 创建一个随机智能体，用于在预填充阶段生成随机动作。
  random_agent = embodied.RandomAgent(train_env.act_space)
  print('预填充--训练--数据集。')
  # 使用随机策略与环境交互，直到训练回放缓冲区达到指定大小。
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
  print('预填充--评估--数据集。')
  # 使用随机策略与环境交互，直到评估回放缓冲区达到指定大小。
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)

  # 将当前性能指标结果添加到日志中。
  logger.add(metrics.result())
  # 写入日志文件。
  logger.write()

  # 创建训练数据集，从训练回放缓冲区中获取数据。
  dataset_train = agent.dataset(train_replay.dataset)
  # 创建评估数据集，从评估回放缓冲区中获取数据。
  dataset_eval = agent.dataset(eval_replay.dataset)

  # 初始化状态列表，以便在训练步骤函数中可写入。
  state = [None]  # 状态列表，初始值为 None
  # 初始化批次列表，以便在训练步骤函数中可写入。
  batch = [None]  # 批次列表，初始值为 None


  def train_step(tran, worker):
    """执行训练步骤。

    根据should_train条件进行多次训练，并同步模型、记录日志等操作。
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

  # 触发训练步骤
  driver_train.on_step(train_step)

  # 初始化检查点，用于模型、经验回放等的保存与加载
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  # 如果指定了检查点路径，则加载模型等数据
  if args.from_checkpoint:
      checkpoint.load(args.from_checkpoint)
  # 加载或保存检查点
  checkpoint.load_or_save()
  # 注册我们刚刚保存了
  should_save(step)

  print('Start training loop.')
  # 定义探索或训练模式下的策略
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  # 定义评估模式下的策略
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  # 主训练循环
  while step < args.steps:
      # 如果需要评估，则执行评估
      if should_eval(step):
          print('Starting evaluation at step', int(step))
          driver_eval.reset()
          driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps), lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
      # 执行训练
      driver_train(policy_train, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
      # 如果需要保存，则保存检查点
      if should_save(step):
          checkpoint.save()
  # 写入日志
  logger.write()
  logger.write()


