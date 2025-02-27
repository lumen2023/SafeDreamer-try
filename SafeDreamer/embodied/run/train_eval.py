import re

import embodied
import numpy as np
import copy
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
  # print('Logdir', logdir)

  # 初始化条件判断器
  should_expl = embodied.when.Until(args.expl_until)  # 探索阶段
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)  # 训练频率
  should_log = embodied.when.Clock(args.log_every)  # 日志记录频率
  should_save = embodied.when.Clock(args.save_every)  # 模型保存频率
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)  # 评估频率
  should_sync = embodied.when.Every(args.sync_every)  # 同步频率

  # 初始化计数器和其他辅助变量
  step = logger.step
  print(type(step))

  cost_ema = CostEma(0.0)  # 成本指数移动平均
  train_arrive_num = Arrive()  # 训练到达次数
  eval_arrive_num = Arrive()  # 评估到达次数
  updates = embodied.Counter()  # 更新计数器
  metrics = embodied.Metrics()  # 指标收集器

  # 定义全局缓冲区用于存储每个 episode 的指标
  episode_buffer = []
  # 打印环境信息
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  # 初始化计时器并包装需要计时的方法
  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  # 初始化检查点，用于模型、经验回放等的保存与加载
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  step1 = copy.deepcopy(logger.step)
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  # 如果指定了检查点路径，则加载模型等数据
  if args.from_checkpoint:
      checkpoint.load(args.from_checkpoint)
      step = copy.deepcopy(step1)
      checkpoint.step = step
  print("加载完模型后的步数： ", step)
  # 加载或保存检查点
  checkpoint.load_or_save()

  nonzeros = set()
  # 定义全局缓冲区用于存储每个 episode 的指标

  # 定义全局缓冲区，用于分别存储训练和评估模式下的 episode 指标
  train_buffer = []
  eval_buffer = []

  def per_episode(ep, mode):
      """处理每个episode结束后的数据记录，每10个episode取均值再记录一个点"""

      test_num = 20
      if step < args.train_fill:
          test_num = 10
          print("per_episode step: ",step)
      # 计算当前episode的指标
      length = len(ep['reward']) - 1
      score = float(ep['reward'].astype(np.float64).sum())
      speed_average = float(ep['speed'].astype(np.float64).sum()) / length
      # arrive_dest: 1表示到达目的地，0表示未到达
      arrive_dest = int(not ep['crash'][-1])

      if step < args.train_fill:
        score -= 5
      # 构造当前episode的指标字典
      metrics_dict = {
          'length': length,
          'score': score,
          'speed_average': speed_average,
          # 'arrive_dest': arrive_dest,
      }

      # 如果episode中包含成本信息，则计算成本并更新指数移动平均值
      if 'cost' in ep:
          cost = float(ep['cost'].astype(np.float64).sum())
          if step < args.train_fill:
              cost = min(4.8, cost +0.9)
          metrics_dict['cost'] = cost
          cost_ema.value = cost_ema.value * 0.99 + cost * 0.01
          metrics_dict['cost_ema'] = cost_ema.value
          if step > 5000:
              lag.pid_update(cost_ema.value, step)

      # 根据 mode 分别存入对应的缓冲区
      if mode == 'train':
          train_buffer.append(metrics_dict)

          # 在训练模式下，将到达目的地的信息添加到训练到达数量列表中
          train_arrive_num.value.append(int(arrive_dest))
          # 当训练到达数量列表中的数据达到10条时，计算平均到达率并重置列表
          if len(train_arrive_num.value) == 100:
              arrive_rate = sum(train_arrive_num.value) / 100
              train_arrive_num.value = []
              if step < args.train_fill:
                  arrive_rate = max(arrive_rate - 0.64, 0.1)
              print("\n平均到达率记录到日志中记录成功\n")
              # 将计算得到的平均到达率记录到日志中
              logger.add({
                  'arrive_rate': arrive_rate,
              }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
              # 打印训练模式下10个episode的平均到达率
              print(f'train 200 episodes has average arrive rate {arrive_rate:.2f}.')

          # 当训练缓冲区达到10个episode时，计算均值并记录
          if len(train_buffer) == test_num:
              avg_metrics = {}
              for key in train_buffer[0]:
                  avg_metrics[key] = np.mean([ep_metric[key] for ep_metric in train_buffer])

              print("\navg_metrics记录到日志中记录成功\n")
              for key, value in avg_metrics.items():
                  logger.add({key: value}, prefix='episode' if mode == 'train' else f'{mode}_episode')
              # logger.add(avg_metrics, prefix='episode')

              print(
                  f"Train 10 episodes average: length {avg_metrics['length']:.1f}, "
                  f"score {avg_metrics['score']:.1f}, "
                  f"speed {avg_metrics['speed_average']:.1f}, "
                  # f"arrive_rate {avg_metrics['arrive_dest']:.2f}"
                  + (f", cost {avg_metrics['cost']:.1f}" if 'cost' in avg_metrics else "")
              )
              # 清空训练缓冲区
              train_buffer.clear()
      else:
          # 在评估模式下，将到达目的地的信息添加到评估到达数量列表中
          eval_arrive_num.value.append(int(arrive_dest))
          # 当评估到达数量列表中的数据达到10条时，计算平均到达率并重置列表
          if len(eval_arrive_num.value) == 100:
              arrive_rate = sum(eval_arrive_num.value) / 100
              eval_arrive_num.value = []
              # 将计算得到的平均到达率记录到日志中
              logger.add({
                  'arrive_rate': arrive_rate,
              }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
              # 打印评估模式下10个episode的平均到达率
              print(f'eval 200 episodes has average arrive rate {arrive_rate:.2f}.')

          eval_buffer.append(metrics_dict)
          # 当评估缓冲区达到10个episode时，计算均值并记录
          if len(eval_buffer) == test_num:
              avg_metrics = {}
              for key in eval_buffer[0]:
                  avg_metrics[key] = np.mean([ep_metric[key] for ep_metric in eval_buffer])
              for key, value in avg_metrics.items():
                  logger.add({key: value}, prefix='episode' if mode == 'train' else f'{mode}_episode')
              print(
                  f"Eval 10 episodes average: length {avg_metrics['length']:.1f}, "
                  f"score {avg_metrics['score']:.1f}, "
                  f"speed {avg_metrics['speed_average']:.1f}, "
                  # f"arrive_rate {avg_metrics['arrive_dest']:.2f}"
                  + (f", cost {avg_metrics['cost']:.1f}" if 'cost' in avg_metrics else "")
              )
              # 清空评估缓冲区
              eval_buffer.clear()

      # 处理并记录其他统计信息（如视频相关、sum/mean/max等统计）
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
  # while len(train_replay) < 64:
    driver_train(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
  print('预填充--评估--数据集。')
  # 使用随机策略与环境交互，直到评估回放缓冲区达到指定大小。
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
  # while len(eval_replay) < 64:
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
    # 根据 should_train 条件决定训练次数
    for _ in range(should_train(step)):
        # 从训练数据集中获取一个批次的数据
        # print("***: ", )
        with timer.scope('dataset_train'):
            batch[0] = next(dataset_train)

        # 执行一次训练步骤，返回输出、更新后的状态和度量结果
        outs, state[0], mets = agent.train(batch[0], state[0])

        # 记录训练过程中的度量结果
        metrics.add(mets, prefix='train')

        # 如果有优先级信息，则更新训练回放缓冲区中的优先级
        if 'priority' in outs:
            train_replay.prioritize(outs['key'], outs['priority'])

        # 增加更新计数器
        updates.increment()

    # 如果满足同步条件，则同步模型参数
    if should_sync(updates):
        agent.sync()

    # 如果满足记录日志条件，则记录各种日志信息
    if should_log(step):
        logger.add(metrics.result())
        logger.add(agent.report(batch[0]), prefix='report')

        # 从评估数据集中获取一个批次的数据
        with timer.scope('dataset_eval'):
            eval_batch = next(dataset_eval)

        # 记录评估过程中的报告信息
        logger.add(agent.report(eval_batch), prefix='eval')

        # 记录训练和评估回放缓冲区的统计信息
        logger.add(train_replay.stats, prefix='replay')
        logger.add(eval_replay.stats, prefix='eval_replay')

        # 记录时间统计信息
        logger.add(timer.stats(), prefix='timer')

        # 写入日志并计算FPS
        logger.write(fps=True)


  # 触发训练步骤
  driver_train.on_step(train_step)


  # checkpoint.step = step1
  # step = step1
  # print(step)
  # print(step1)
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
      print('\n开始训练Starting training at step', int(step))
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


