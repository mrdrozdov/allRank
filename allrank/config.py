import json
from collections import defaultdict
from typing import Dict, List, Optional

from attr import attrib, attrs


@attrs
class TransformerConfig:
    N = attrib(type=int)
    d_ff = attrib(type=int)
    h = attrib(type=int)
    positional_encoding = attrib(type=dict)
    dropout = attrib(type=float)


@attrs
class FCConfig:
    sizes = attrib(type=List[int])
    input_norm = attrib(type=bool)
    activation = attrib(type=str)
    dropout = attrib(type=float)
    embed_size = attrib(type=int)
    embed_x_tgt = attrib(type=bool, default=False)
    embed_q_src = attrib(type=bool, default=False)


@attrs
class PostModelConfig:
    d_output = attrib(type=int)
    output_activation = attrib(type=str)


@attrs
class ModelConfig:
    fc_model = attrib(type=FCConfig)
    transformer = attrib(type=TransformerConfig)
    post_model = attrib(type=PostModelConfig)


@attrs
class DstoreConfig:
    path = attrib(type=str)
    dstore_size = attrib(type=int)
    vec_size = attrib(type=int)
    enabled = attrib(type=bool, default=False)
    load_in_collate = attrib(type=bool, default=False)
    load_in_main_loop = attrib(type=bool, default=False)
    main_loop_batch = attrib(type=int, default=10)
    load_xb = attrib(type=bool, default=False)
    prefetch = attrib(type=bool, default=False)


@attrs
class PositionalEncoding:
    strategy = attrib(type=str)
    max_indices = attrib(type=int)


@attrs
class DataConfig:
    path = attrib(type=str)
    num_workers = attrib(type=int)
    batch_size = attrib(type=int)
    slate_length = attrib(type=int)
    validation_ds_role = attrib(type=str)


@attrs
class LogConfig:
    save_every_epoch = attrib(type=bool, default=False)


@attrs
class TrainingConfig:
    epochs = attrib(type=int)
    gradient_clipping_norm = attrib(type=float)
    early_stopping_patience = attrib(type=int, default=0)


@attrs
class NameArgsConfig:
    name = attrib(type=str)
    args = attrib(type=dict)


@attrs
class Config:
    model = attrib(type=ModelConfig)
    dstore = attrib(type=DstoreConfig)
    log = attrib(type=LogConfig)
    data = attrib(type=DataConfig)
    optimizer = attrib(type=NameArgsConfig)
    training = attrib(type=TrainingConfig)
    loss = attrib(type=NameArgsConfig)
    metrics = attrib(type=Dict[str, List[int]])
    lr_scheduler = attrib(type=NameArgsConfig)
    val_metric = attrib(type=str, default=None)
    expected_metrics = attrib(type=Dict[str, Dict[str, float]], default={})
    detect_anomaly = attrib(type=bool, default=False)
    click_model = attrib(type=Optional[NameArgsConfig], default=None)

    @classmethod
    def from_json(cls, config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
            return Config.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        config["model"] = ModelConfig(**config["model"])
        if config["model"].transformer:
            config["model"].transformer = TransformerConfig(**config["model"].transformer)
            if config["model"].transformer.positional_encoding:
                config["model"].transformer.positional_encoding = PositionalEncoding(
                    **config["model"].transformer.positional_encoding)
        config["data"] = DataConfig(**config["data"])
        config["optimizer"] = NameArgsConfig(**config["optimizer"])
        config["training"] = TrainingConfig(**config["training"])
        config["metrics"] = cls._parse_metrics(config["metrics"])
        config["lr_scheduler"] = NameArgsConfig(**config["lr_scheduler"])
        config["loss"] = NameArgsConfig(**config["loss"])
        if "click_model" in config.keys():
            config["click_model"] = NameArgsConfig(**config["click_model"])
        return cls(**config)

    @staticmethod
    def _parse_metrics(metrics):
        metrics_dict = defaultdict(list)  # type: Dict[str, list]
        for metric_string in metrics:
            try:
                name, at = metric_string.split("_")
                metrics_dict[name].append(int(at))
            except (ValueError, TypeError):
                raise MetricConfigError(
                    metric_string,
                    "Wrong formatting of metric in config. Expected format: <name>_<at> where name is valid metric name and at is and int")
        return metrics_dict


class MetricConfigError(Exception):
    pass
