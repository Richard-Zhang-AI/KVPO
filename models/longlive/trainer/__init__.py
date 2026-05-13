from importlib import import_module

__all__ = [
    "ScoreDistillationTrainer",
    "KVPOTrainer",
]

_EXPORTS = {
    "ScoreDistillationTrainer": (".distillation", "Trainer"),
    "KVPOTrainer": (".kvpo", "KVPOTrainer"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
