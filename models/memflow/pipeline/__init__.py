from importlib import import_module

__all__ = [
    "CausalInferencePipeline",
    "InteractiveCausalInferencePipeline",
    "SelfForcingTrainingPipeline",
    "StreamingTrainingPipeline",
    "StreamingSwitchTrainingPipeline",
    "SwitchCausalInferencePipeline",
]

_EXPORTS = {
    "CausalInferencePipeline": (".causal_inference", "CausalInferencePipeline"),
    "InteractiveCausalInferencePipeline": (".interactive_causal_inference", "InteractiveCausalInferencePipeline"),
    "SelfForcingTrainingPipeline": (".self_forcing_training", "SelfForcingTrainingPipeline"),
    "StreamingTrainingPipeline": (".streaming_training", "StreamingTrainingPipeline"),
    "StreamingSwitchTrainingPipeline": (".streaming_switch_training", "StreamingSwitchTrainingPipeline"),
    "SwitchCausalInferencePipeline": (".switch_causal_inference", "SwitchCausalInferencePipeline"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
