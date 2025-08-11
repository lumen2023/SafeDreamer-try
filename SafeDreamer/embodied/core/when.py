import time


class Every:

  def __init__(self, every, initial=True):
    self._every = every
    self._initial = initial
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    if self._prev is None:
      self._prev = (step // self._every) * self._every
      return self._initial
    if step >= self._prev + self._every:
      self._prev += self._every
      return True
    return False


class Ratio:

  def __init__(self, ratio):
    assert ratio >= 0, ratio
    self._ratio = ratio
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:
  """
  该类用于表示一个“直到”条件判断器，常用于控制某些操作在达到指定步数之前持续执行。

  Attributes:
    _until (int): 指定的停止步数。当 step >= _until 时，返回 False，否则返回 True。
  """

  def __init__(self, until):
    """
    初始化 Until 实例。

    Args:
      until (int): 指定的停止步数。如果为 None 或 0，则默认始终返回 True。
    """
    self._until = until

  def __call__(self, step):
    """
    根据当前步数决定是否继续执行。

    Args:
      step (int): 当前步数。

    Returns:
      bool: 如果 step < _until 返回 True，否则返回 False。
    """
    step = int(step)
    if not self._until:
      return True
    return step < self._until


class Clock:

  def __init__(self, every):
    self._every = every
    self._prev = None

  def __call__(self, step=None):
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    now = time.time()
    if self._prev is None:
      self._prev = now
      return True
    if now >= self._prev + self._every:
      # self._prev += self._every
      self._prev = now
      return True
    return False
