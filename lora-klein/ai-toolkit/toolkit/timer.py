import time
from collections import OrderedDict, deque
import sys
import os

# check if is ui process will have IS_AI_TOOLKIT_UI in env
is_ui = os.environ.get("IS_AI_TOOLKIT_UI", "0") == "1"

class Timer:
    def __init__(self, name='Timer', max_buffer=10):
        self.name = name
        self.max_buffer = max_buffer
        self.timers = OrderedDict()
        self.total_elapsed = OrderedDict()
        self.total_counts = OrderedDict()
        self.active_timers = {}
        self.current_timer = None  # Used for the context manager functionality
        self._after_print_hooks = []

    def start(self, timer_name):
        if timer_name not in self.timers:
            self.timers[timer_name] = deque(maxlen=self.max_buffer)
        if timer_name not in self.total_elapsed:
            self.total_elapsed[timer_name] = 0.0
        if timer_name not in self.total_counts:
            self.total_counts[timer_name] = 0
        self.active_timers[timer_name] = time.time()

    def cancel(self, timer_name):
        """Cancel an active timer."""
        if timer_name in self.active_timers:
            del self.active_timers[timer_name]

    def stop(self, timer_name):
        if timer_name not in self.active_timers:
            raise ValueError(f"Timer '{timer_name}' was not started!")

        elapsed_time = time.time() - self.active_timers[timer_name]
        self.timers[timer_name].append(elapsed_time)
        self.total_elapsed[timer_name] += elapsed_time
        self.total_counts[timer_name] += 1

        # Clean up active timers
        del self.active_timers[timer_name]

        # Check if this timer's buffer exceeds max_buffer and remove the oldest if it does
        if len(self.timers[timer_name]) > self.max_buffer:
            self.timers[timer_name].popleft()

    def add_after_print_hook(self, hook):
        self._after_print_hooks.append(hook)

    def print(self):
        if not is_ui:
            print(f"\nTimer '{self.name}':")
        timing_dict = {}
        # sort by longest at top
        for timer_name, timings in sorted(self.timers.items(), key=lambda x: sum(x[1]), reverse=True):
            avg_time = sum(timings) / len(timings)
            
            if not is_ui:
                print(f" - {avg_time:.4f}s avg - {timer_name}, num = {len(timings)}")
            timing_dict[timer_name] = avg_time

        for hook in self._after_print_hooks:
            hook(timing_dict)
        if not is_ui:
            print('')

    def snapshot_totals(self):
        snapshot = {}
        for timer_name in set(self.total_elapsed.keys()) | set(self.total_counts.keys()):
            snapshot[timer_name] = (
                int(self.total_counts.get(timer_name, 0)),
                float(self.total_elapsed.get(timer_name, 0.0)),
            )
        return snapshot

    def get_totals_since(self, snapshot):
        deltas = OrderedDict()
        timer_names = list(self.total_elapsed.keys())
        for timer_name in timer_names:
            prev_count, prev_total = snapshot.get(timer_name, (0, 0.0))
            count_delta = int(self.total_counts.get(timer_name, 0)) - int(prev_count)
            total_delta = float(self.total_elapsed.get(timer_name, 0.0)) - float(prev_total)
            if count_delta <= 0 and total_delta <= 0.0:
                continue
            total_delta = max(0.0, total_delta)
            deltas[timer_name] = {
                "count": max(0, count_delta),
                "total": total_delta,
                "avg": total_delta / max(1, count_delta),
            }
        return deltas

    def reset(self):
        self.timers.clear()
        self.total_elapsed.clear()
        self.total_counts.clear()
        self.active_timers.clear()

    def __call__(self, timer_name):
        """Enable the use of the Timer class as a context manager."""
        self.current_timer = timer_name
        self.start(timer_name)
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # No exceptions, stop the timer normally
            self.stop(self.current_timer)
        else:
            # There was an exception, cancel the timer
            self.cancel(self.current_timer)
