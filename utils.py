from typing import Callable
from modules import devices, shared


def safe_import(import_name: str, pkg_name: str | None = None):
    try:
        __import__(import_name)
    except Exception:
        pkg_name = pkg_name or import_name
        import pip
        if hasattr(pip, 'main'):
            pip.main(['install', pkg_name])
        else:
            pip._internal.main(['install', pkg_name])
        __import__(import_name)


class CondCache:
    def __init__(self, function: Callable):
        self.cond = (None, None, None)
        self.func = function

    def get_cond(self, prompt: [str], steps: int):
        cond, cached_prompt, cached_steps = self.cond
        if cond is not None and (prompt, steps) == (cached_prompt, cached_steps):
            return cond

        with devices.autocast():
            cond = self.func(shared.sd_model, prompt, steps)

        self.cond = (cond, prompt, steps)
        return cond

