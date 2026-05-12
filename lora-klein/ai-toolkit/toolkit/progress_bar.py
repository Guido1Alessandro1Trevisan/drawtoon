from tqdm import tqdm
import time
import sys


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        output_file = kwargs.get("file", None) or sys.stderr
        is_interactive = bool(getattr(output_file, "isatty", lambda: False)())
        if not is_interactive:
            kwargs.setdefault("dynamic_ncols", False)
            kwargs.setdefault("mininterval", 1.0)
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()
        self.is_interactive = is_interactive

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)

    def set_postfix_str(self, s="", refresh=True):
        if not self.is_interactive:
            refresh = False
        super().set_postfix_str(s=s, refresh=refresh)
