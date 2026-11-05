"""
noesis/config.py
================
Configuration centrale de Noesis (bibliothèque standard uniquement).
Aucun module n'embarque de constante magique : tout hyper-paramètre vit ici.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Union

CONFIG_VERSION = 1


class Op(enum.IntEnum):
    """Opérations épistémiques que le contrôleur choisit à chaque pas."""

    READ = 0       # lire une fenêtre d'octets (branche, offset, longueur, vue)
    THINK = 1      # relaxer la dynamique, sans entrée ni sortie
    IMAGINE = 2    # do() interne sur une unité ou une poignée ; rien n'est envoyé au monde
    INTERVENE = 3  # do() réel via une poignée, puis relecture de la nouvelle branche
    TRANSFORM = 4  # relecture transformée de la dernière fenêtre (sonde d'invariance)
    REVISE = 5     # rafale de plasticité neuromodulée guidée par la surprise
    COMMIT = 6     # clore l'épisode


N_OPS: int = len(Op)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Configuration invalide : {msg}")


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


class _Section:
    """Base commune : chaque section se valide à la construction."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        return None


# --------------------------------------------------------------------------- substrat

@dataclass
class GraphConfig(_Section):
    """Pools de nœuds/arêtes à capacité fixe (la topologie change par masques)."""

    n_max: int = 384                 # capacité du pool de nœuds
    n_init: int = 192                # nœuds vivants au départ (capteurs + poignées + cachés)
    n_sensor: int = 32               # ports sensoriels : indices [0, n_sensor)
    n_handle: int = 8                # ports de poignées : [n_sensor, n_sensor + n_handle)
    e_max: int = 12288               # capacité du pool d'arêtes
    init_out_degree: int = 8
    max_in_degree: int = 64
    forbid_reciprocal_init: bool = True
    allow_self_loops: bool = False
    d_state: int = 64
    d_trait: int = 16                # identité/adresse lente de chaque nœud
    n_relation: int = 8              # K canaux relationnels (matrices de base)
    d_rel_rank: int = 8              # rang r de chaque matrice de base B_k = U_k V_k^T
    init_weight_std: float = 0.1
    w_max: float = 3.0
    gate_beta: float = 2.0 / 3.0     # hard-concrete (régularisation L0 des arêtes)
    gate_gamma: float = -0.1
    gate_zeta: float = 1.1
    gate_init_keep: float = 0.9

    def validate(self) -> None:
        _require(self.n_sensor >= 1, "graph.n_sensor >= 1")
        _require(self.n_handle >= 0, "graph.n_handle >= 0")
        _require(self.n_init - self.n_sensor - self.n_handle >= 4,
                 "graph.n_init doit laisser au moins 4 nœuds cachés")
        _require(self.n_init <= self.n_max, "graph.n_init <= graph.n_max")
        _require(self.init_out_degree >= 1, "graph.init_out_degree >= 1")
        _require(self.n_init * self.init_out_degree <= int(0.8 * self.e_max),
                 "graph.e_max doit valoir >= 1.25 * n_init * init_out_degree")
        _require(self.max_in_degree >= 1, "graph.max_in_degree >= 1")
        _require(self.d_state >= 8 and self.d_trait >= 1, "graph.d_state >= 8 et d_trait >= 1")
        _require(self.n_relation >= 1 and self.d_rel_rank >= 1, "graph.n_relation, d_rel_rank >= 1")
        _require(self.init_weight_std > 0 and self.w_max > 0, "graph.init_weight_std, w_max > 0")
        _require(self.gate_gamma < 0.0 < self.gate_beta and self.gate_zeta > 1.0,
                 "hard-concrete : gamma < 0, beta > 0, zeta > 1")
        _require(0.0 < self.gate_init_keep < 1.0, "graph.gate_init_keep dans ]0, 1[")


@dataclass
class DynamicsConfig(_Section):
    settle_steps: int = 4            # pas de relaxation par lecture/pensée
    imagine_steps: int = 6           # pas de relaxation pour un do() interne
    revise_steps: int = 2            # pas de plasticité forcée pour REVISE
    d_plast: int = 8                 # dimension des traces pré/post (poids rapides)
    fast_decay: float = 0.9
    fast_clip: float = 2.0
    plast_lr_init: float = 0.05
    d_mod: int = 16                  # état neuromodulateur global
    noise_std: float = 0.0           # bruit de dynamique (modulé par le neuromodulateur)
    edge_dropout: float = 0.0
    degree_norm: str = "sqrt"        # "none" | "sqrt" | "mean"
    film_scale: float = 0.5          # amplitude de la modulation FiLM par les traits de nœud
    min_timescale: float = 0.1       # borne basse du taux de mise à jour par nœud

    def validate(self) -> None:
        _require(min(self.settle_steps, self.imagine_steps, self.revise_steps) >= 1,
                 "dynamics.*_steps >= 1")
        _require(self.d_plast >= 1 and self.d_mod >= 4, "dynamics.d_plast >= 1, d_mod >= 4")
        _require(0.0 <= self.fast_decay < 1.0, "dynamics.fast_decay dans [0, 1[")
        _require(self.fast_clip > 0 and self.plast_lr_init > 0, "dynamics.fast_clip, plast_lr_init > 0")
        _require(self.noise_std >= 0 and 0.0 <= self.edge_dropout < 1.0, "dynamics.noise/dropout")
        _require(self.degree_norm in ("none", "sqrt", "mean"), "dynamics.degree_norm invalide")
        _require(0.0 < self.min_timescale <= 1.0 and self.film_scale >= 0.0, "dynamics.timescale/film")


@dataclass
class StructureConfig(_Section):
    """Plasticité structurelle (hors autograd, entre deux mises à jour)."""

    enabled: bool = True
    every_updates: int = 250
    util_ema: float = 0.99
    protect_updates: int = 500       # âge minimal avant qu'un élément neuf soit élagable
    prune_threshold: float = 1e-3
    prune_max_frac: float = 0.05
    grow_max_frac: float = 0.05
    grow_candidates: int = 4096
    grow_init_std: float = 0.02
    node_birth_max: int = 4
    node_birth_quantile: float = 0.95
    node_death_threshold: float = 1e-4
    node_death_max: int = 4
    reset_optimizer_state: bool = True

    def validate(self) -> None:
        _require(self.every_updates >= 1 and self.protect_updates >= 0, "structure.every/protect")
        _require(0.0 < self.util_ema < 1.0, "structure.util_ema dans ]0, 1[")
        _require(0.0 <= self.prune_max_frac <= 0.5 and 0.0 <= self.grow_max_frac <= 0.5,
                 "structure.*_max_frac dans [0, 0.5]")
        _require(self.grow_candidates >= 1 and self.grow_init_std > 0, "structure.grow_*")
        _require(self.node_birth_max >= 0 and self.node_death_max >= 0, "structure.node_*_max >= 0")
        _require(0.5 <= self.node_birth_quantile < 1.0, "structure.node_birth_quantile")


# --------------------------------------------------------------------------- perception

@dataclass
class PerceptionConfig(_Section):
    d_byte: int = 32
    d_channels: int = 64
    n_blocks: int = 6
    kernel: int = 5
    dilation_base: int = 2
    n_patches: int = 32              # bins par lecture (+1 jeton de croquis global)
    min_read: int = 64               # longueurs de lecture : min_read * 2^i <= max_read
    max_read: int = 4096
    n_pos_bands: int = 6
    sketch_lag_bins: int = 32        # bins log-espacés de l'autocorrélation
    d_patch: int = 64
    d_route: int = 32
    route_topk: int = 2
    route_temp: float = 1.0
    transform_max_frac: float = 0.25 # fraction maximale d'octets touchés par une transformation
    dropout: float = 0.0
    pad_id: int = 256

    def validate(self) -> None:
        _require(min(self.d_byte, self.d_route) >= 4 and self.d_channels >= 8 and self.d_patch >= 8,
                 "perception.d_* trop petits")
        _require(self.n_blocks >= 1 and self.kernel >= 3 and self.kernel % 2 == 1,
                 "perception.n_blocks >= 1, kernel impair >= 3")
        _require(self.dilation_base >= 1 and self.n_patches >= 1, "perception.dilation_base, n_patches")
        _require(_is_pow2(self.min_read) and _is_pow2(self.max_read) and self.min_read <= self.max_read,
                 "perception.min_read/max_read : puissances de 2, min <= max")
        _require(self.n_pos_bands >= 1 and self.sketch_lag_bins >= 4, "perception.n_pos_bands, sketch_lag_bins")
        _require(self.route_topk >= 1 and self.route_temp > 0, "perception.route_*")
        _require(0.0 < self.transform_max_frac <= 1.0, "perception.transform_max_frac")
        _require(0.0 <= self.dropout < 1.0, "perception.dropout")
        _require(self.pad_id == 256, "perception.pad_id doit valoir 256")


# --------------------------------------------------------------------------- compréhension

@dataclass
class WorkspaceConfig(_Section):
    n_units: int = 24                # slots d'unités M
    d_unit: int = 64
    d_mech: int = 32
    d_pair: int = 64
    d_cond: int = 32
    kmeans_iters: int = 2
    pool_temp: float = 1.0
    patch_bank: int = 512            # patches mémorisés au cours de l'épisode
    effect_eps: float = 1e-4
    probe_values: int = 2            # valeurs testées en parallèle par un do() interne

    def validate(self) -> None:
        _require(self.n_units >= 2, "workspace.n_units >= 2")
        _require(min(self.d_unit, self.d_mech, self.d_pair, self.d_cond) >= 4, "workspace.d_* >= 4")
        _require(self.kmeans_iters >= 1 and self.pool_temp > 0, "workspace.kmeans_iters, pool_temp")
        _require(self.patch_bank >= 8 and self.effect_eps > 0 and self.probe_values >= 1,
                 "workspace.patch_bank, effect_eps, probe_values")


@dataclass
class AnswerConfig(_Section):
    d_query: int = 64
    d_decoder: int = 128
    reason_steps: int = 2            # relaxations du graphe avec la requête injectée
    min_logstd: float = -5.0
    max_logstd: float = 3.0

    def validate(self) -> None:
        _require(self.d_query >= 8 and self.d_decoder >= 8 and self.reason_steps >= 1, "answer.*")
        _require(self.min_logstd < self.max_logstd, "answer.min_logstd < max_logstd")


# --------------------------------------------------------------------------- apprentissage

@dataclass
class ObjectiveConfig(_Section):
    d_latent: int = 32
    nce_temperature: float = 0.1
    nce_learnable: bool = True
    nce_min_temperature: float = 0.01
    ib_beta: float = 1e-2
    ib_warmup_updates: int = 4000
    ib_free_bits: float = 0.02
    vrex_min_envs: int = 2
    w_contrastive: float = 1.0
    w_ib: float = 1.0
    w_interventional: float = 1.0
    w_hypothesis: float = 0.5
    w_vrex: float = 0.1
    w_locality: float = 0.05
    w_calibration: float = 0.5
    w_balance: float = 0.01
    w_l0: float = 1e-5
    w_stability: float = 1e-3
    w_policy: float = 1.0

    def validate(self) -> None:
        _require(self.d_latent >= 2, "objectives.d_latent >= 2")
        _require(self.nce_temperature > self.nce_min_temperature > 0, "objectives.nce_temperature")
        _require(self.ib_beta >= 0 and self.ib_warmup_updates >= 0 and self.ib_free_bits >= 0,
                 "objectives.ib_*")
        _require(self.vrex_min_envs >= 2, "objectives.vrex_min_envs >= 2")
        for name in ("w_contrastive", "w_ib", "w_interventional", "w_hypothesis", "w_vrex", "w_locality",
                     "w_calibration", "w_balance", "w_l0", "w_stability", "w_policy"):
            _require(getattr(self, name) >= 0.0, f"objectives.{name} >= 0")


@dataclass
class ControllerConfig(_Section):
    d_ctrl: int = 128
    d_action_value: int = 8          # dimension des arguments continus d'action
    n_transforms: int = 6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    shaping_coef: float = 0.1        # shaping par gain d'information (potentiel)
    action_logstd_min: float = -3.0
    action_logstd_max: float = 0.5
    # coût par opération, dans l'ordre de ``Op`` : READ, THINK, IMAGINE, INTERVENE, TRANSFORM, REVISE, COMMIT
    op_costs: Tuple[float, ...] = (0.010, 0.002, 0.004, 0.030, 0.008, 0.004, 0.0)
    reward_w_justesse: float = 1.0
    reward_w_couverture: float = 1.0
    reward_w_calibration: float = 1.0

    def validate(self) -> None:
        _require(self.d_ctrl >= 16 and self.d_action_value >= 1 and self.n_transforms >= 1, "controller.d_*")
        _require(0.0 < self.gamma <= 1.0 and 0.0 <= self.gae_lambda <= 1.0, "controller.gamma, gae_lambda")
        _require(self.entropy_coef >= 0 and self.value_coef >= 0 and self.shaping_coef >= 0,
                 "controller.*_coef >= 0")
        _require(self.action_logstd_min < self.action_logstd_max, "controller.action_logstd_*")
        _require(len(self.op_costs) == N_OPS and all(c >= 0 for c in self.op_costs),
                 f"controller.op_costs : {N_OPS} valeurs >= 0")
        weights = (self.reward_w_justesse, self.reward_w_couverture, self.reward_w_calibration)
        _require(all(w >= 0 for w in weights) and sum(weights) > 0, "controller.reward_w_*")


# --------------------------------------------------------------------------- interface & entraînement

@dataclass
class InterfaceConfig(_Section):
    """Bornes dures de la connexion au monde (dimensionnent les tenseurs)."""

    max_handles: int = 8
    max_handle_dim: int = 8
    max_branches: int = 8
    max_views: int = 4
    max_steps: int = 32
    max_queries: int = 32
    max_candidates: int = 8
    max_prompt_bytes: int = 1024
    max_answer_bytes: int = 256

    def validate(self) -> None:
        for name in ("max_handles", "max_handle_dim", "max_steps", "max_queries", "max_candidates",
                     "max_prompt_bytes", "max_answer_bytes"):
            _require(getattr(self, name) >= 1, f"interface.{name} >= 1")
        _require(self.max_branches >= 2 and self.max_views >= 1, "interface.max_branches >= 2, max_views >= 1")


@dataclass
class TrainingConfig(_Section):
    seed: int = 0
    device: str = "auto"
    amp: bool = False
    amp_dtype: str = "bfloat16"
    batch_episodes: int = 8
    n_updates: int = 100_000
    lr: float = 3e-4
    lr_final_frac: float = 0.05
    warmup_updates: int = 500
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    grad_clip: float = 1.0
    detach_every: int = 0            # 0 = BPTT complet sur l'épisode ; sinon troncature
    ema_decay: float = 0.999
    eval_every: int = 1000
    eval_episodes: int = 64
    ckpt_every: int = 1000
    ckpt_dir: str = "runs/noesis"
    keep_last: int = 3
    log_every: int = 20
    train_split: str = "train"
    val_split: str = "val"
    ood_split: str = "ood"

    def validate(self) -> None:
        _require(self.device in ("auto", "cpu", "cuda", "mps") or self.device.startswith("cuda:"),
                 "training.device invalide")
        _require(self.amp_dtype in ("bfloat16", "float16"), "training.amp_dtype invalide")
        _require(self.batch_episodes >= 1 and self.n_updates >= 1, "training.batch_episodes, n_updates >= 1")
        _require(self.lr > 0 and 0.0 <= self.lr_final_frac <= 1.0 and self.warmup_updates >= 0,
                 "training.lr, lr_final_frac, warmup_updates")
        _require(self.weight_decay >= 0 and self.grad_clip > 0, "training.weight_decay, grad_clip")
        _require(len(self.betas) == 2 and all(0.0 <= b < 1.0 for b in self.betas), "training.betas")
        _require(self.detach_every >= 0 and 0.0 <= self.ema_decay < 1.0, "training.detach_every, ema_decay")
        _require(min(self.eval_every, self.eval_episodes, self.ckpt_every, self.keep_last, self.log_every) >= 1,
                 "training.eval_*, ckpt_*, keep_last, log_every >= 1")
        for s in (self.train_split, self.val_split, self.ood_split):
            _require(s in ("train", "val", "ood"), f"training : split inconnu {s!r}")


# --------------------------------------------------------------------------- racine

def _build(cls: type, data: Dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Champs inconnus pour {cls.__name__} : {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp):
            kwargs[name] = _build(tp, value)
        elif typing.get_origin(tp) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclass
class NoesisConfig(_Section):
    graph: GraphConfig = field(default_factory=GraphConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    objectives: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    version: int = CONFIG_VERSION

    def validate(self) -> None:
        for section in (self.graph, self.dynamics, self.structure, self.perception, self.workspace,
                        self.answer, self.objectives, self.controller, self.interface, self.training):
            section.validate()
        _require(self.version == CONFIG_VERSION, f"version {self.version} != {CONFIG_VERSION}")
        _require(self.graph.n_handle == self.interface.max_handles,
                 "graph.n_handle doit égaler interface.max_handles")
        _require(self.controller.d_action_value >= self.interface.max_handle_dim,
                 "controller.d_action_value >= interface.max_handle_dim")
        _require(self.workspace.patch_bank >= self.perception.n_patches + 1,
                 "workspace.patch_bank >= perception.n_patches + 1")

    # -- utilitaires ---------------------------------------------------------------------------
    def read_lengths(self) -> Tuple[int, ...]:
        """Longueurs de lecture proposables au contrôleur (puissances de 2)."""
        out, n = [], self.perception.min_read
        while n <= self.perception.max_read:
            out.append(n)
            n *= 2
        return tuple(out)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoesisConfig":
        return _build(cls, data)

    def replace(self, **overrides: Any) -> "NoesisConfig":
        """cfg.replace(**{"graph.n_max": 512, "training.lr": 1e-4})"""
        d = self.to_dict()
        for key, value in overrides.items():
            *path, last = key.split(".")
            node = d
            for p in path:
                node = node[p]
            if last not in node:
                raise KeyError(f"Clé de configuration inconnue : {key}")
            node[last] = value
        return NoesisConfig.from_dict(d)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NoesisConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))