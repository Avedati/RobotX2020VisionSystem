import time

global enable_timer
global time_table

enable_timer = False
time_table = {}

class Timer(object):
  def __init__(self, name):
    if not enable_timer:
      return
    self.name = name
    self.start_time = time.time()
    time_info = time_table.get(name, (0, 0))

  def __del__(self):
    if not enable_timer:
      return
    call_time = time.time() - self.start_time
    time_info = time_table.get(self.name, (0, 0))
    total_time = time_info[0] + call_time
    num_calls = time_info[1] + 1
    avg_time = total_time / num_calls

    time_table[self.name] = (total_time, num_calls)

    if num_calls % 50 == 0:
      print('Timer ', self.name,
            ': Total time (sec): ', total_time,
            ', Avg time (msec): ', avg_time * 1000,
            ', FPS: ', 1.0 / avg_time)
