"""Configuration management for MPDR experiments."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class DataConfig:
    data_path: str = "/path/to/wcsim_numpy"
    train_mode: str = "background"
    test_mode: str = "mixed"
    feature_mode: str = "no_charge"
    time_range: Optional[tuple] = None
    augmentations: bool = True
    batch_size: int = 512
    num_workers: int = 16
    train_split: float = 0.8


@dataclass
class ModelConfig:
    # Encoder
    encoder_latent_dim: int = 128
    encoder_num_heads: int = 8
    encoder_num_layers: int = 4
    encoder_dropout: float = 0.0
    encoder_energy_mlp_hidden: Optional[int] = None
    # Decoder
    decoder_seed_dim: int = 3
    decoder_num_heads: int = 8
    decoder_num_layers: int = 3
    decoder_dropout: float = 0.0
    decoder_memory_tokens: int = 4
    # Energy network
    energy_net_type: str = "transformer"
    energy_hidden_dim: int = 128
    energy_num_heads: int = 8
    energy_num_layers: int = 4
    energy_dropout: float = 0.0
    # Autoencoder settings
    spherical: bool = True
    encoding_noise: Optional[float] = None
    ae_loss: str = "chamfer"
    max_output_points: int = 192
    repulsion_weight: float = 0.02


@dataclass
class MPDRConfig:
    variant: str = "simple"
    physics_noise_ratio: float = 0.3
    alphas: List[float] = field(default_factory=lambda: [5.0, 30.0, 120.0])
    # Projection
    proj_mode: str = "uniform"
    proj_noise_start: float = 0.06
    proj_noise_end: float = 0.15
    proj_const: float = 0.1
    proj_const_omi: Optional[float] = 0.1
    proj_dist: str = "geodesic"
    # On-manifold initialisation
    mcmc_n_step_omi: int = 2
    mcmc_stepsize_omi: float = 0.01
    mcmc_noise_omi: float = 0.01
    mcmc_normalize_omi: bool = False
    mh_omi: bool = False
    # Off-manifold recovery
    mcmc_custom_stepsize: bool = False
    mcmc_n_step_x: int = 10
    mcmc_stepsize_x: float = 80.0
    mcmc_noise_x: float = 0.02
    mcmc_bound_x: Optional[tuple] = None
    mh_x: bool = False
    # Gradient clipping
    grad_clip_omi: Optional[float] = 1.0
    grad_clip_off: Optional[float] = 5.0
    # Temperature
    temperature: float = 0.8
    temperature_omi: float = 1.0
    # Regularisation
    gamma_vx: Optional[float] = None
    gamma_neg_recon: Optional[float] = None
    l2_norm_reg_netx: Optional[float] = 1e-4
    lambda_center: float = 1e-4
    tau: float = 10.0
    energy_method: Optional[str] = None


@dataclass
class TrainingConfig:
    phase1_epochs: int = 200
    phase2_epochs: int = 50
    alphas: List[float] = field(default_factory=lambda: [5.0, 30.0, 120.0])
    # Optimisation
    ae_lr: float = 1e-4
    ae_weight_decay: float = 1e-4
    energy_lr: float = 1e-4
    energy_weight_decay: float = 0.0
    clip_grad: Optional[float] = 1.0
    # Scheduler
    use_scheduler: bool = True
    use_warmup: bool = True
    warmup_epochs: int = 5
    early_stopping_patience: Optional[int] = 50
    lr_decay_epochs: List[int] = field(default_factory=list)
    lr_decay_factor: float = 0.2
    # Evaluation
    eval_every: int = 5
    save_every: int = 10
    # Device
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str = "mpdr_hk_trigger"
    output_dir: str = "experiments"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mpdr: MPDRConfig = field(default_factory=MPDRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        feature_dims = {
            "all": 5, "no_time": 4, "no_charge": 4, "no_time_no_charge": 3,
        }
        self.input_dim = feature_dims[self.data.feature_mode]

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert_tuples(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _convert_tuples(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert_tuples(i) for i in obj]
            return obj

        with open(path, "w") as f:
            yaml.dump(_convert_tuples(self.to_dict()), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get("name", "mpdr_hk_trigger"),
            output_dir=d.get("output_dir", "experiments"),
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            mpdr=MPDRConfig(**d.get("mpdr", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def get_default(cls, variant="simple"):
        config = cls()
        config.mpdr.variant = variant
        config.__post_init__()
        return config


def create_mpdr_simple_config():
    """MPDR-S (background mode) defaults."""
    config = ExperimentConfig()
    config.name = "mpdr_simple_hk_trigger"
    config.data.train_mode = "background"
    config.mpdr.variant = "simple"
    config.mpdr.gamma_vx = 1e-4
    config.training.energy_lr = 5e-5
    config.__post_init__()
    return config


def create_mpdr_simple_signal_config():
    """MPDR-S (signal mode) defaults."""
    config = ExperimentConfig()
    config.name = "mpdr_simple_signal_hk_trigger"
    config.data.train_mode = "signal"
    config.mpdr.variant = "simple"
    config.mpdr.proj_noise_start = 0.4
    config.mpdr.proj_noise_end = 0.9
    config.mpdr.mcmc_n_step_omi = 0
    config.mpdr.proj_const = 0
    config.mpdr.mcmc_n_step_x = 5
    config.mpdr.mcmc_noise_x = 0.02
    config.mpdr.mcmc_stepsize_x = 160
    config.mpdr.mcmc_custom_stepsize = False
    config.mpdr.gamma_vx = None
    config.training.energy_lr = 5e-5
    config.__post_init__()
    return config


def create_mpdr_recovery_config():
    """MPDR-R (background mode) defaults."""
    config = ExperimentConfig()
    config.name = "mpdr_recovery_hk_trigger"
    config.data.train_mode = "background"
    config.mpdr.variant = "recovery"
    config.training.energy_lr = 1e-5
    config.__post_init__()
    return config


def create_mpdr_recovery_signal_config():
    """MPDR-R (signal mode) defaults."""
    config = ExperimentConfig()
    config.name = "mpdr_recovery_signal_hk_trigger"
    config.data.train_mode = "signal"
    config.mpdr.variant = "recovery"
    config.training.energy_lr = 1e-5
    config.__post_init__()
    return config


if __name__ == "__main__":
    Path("configs").mkdir(exist_ok=True)
    for fn, name in [
        (create_mpdr_simple_config, "mpdr_simple"),
        (create_mpdr_simple_signal_config, "mpdr_simple_signal"),
        (create_mpdr_recovery_config, "mpdr_recovery"),
        (create_mpdr_recovery_signal_config, "mpdr_recovery_signal"),
    ]:
        cfg = fn()
        cfg.save(f"configs/{name}.yaml")
        print(f"Saved {name} config")
