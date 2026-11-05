"""
noesis/api.py
=============
Contrat monde <-> réseau. Pur Python + NumPy : implémenter un monde ne demande pas PyTorch.

Règles du contrat
-----------------
1. Perception = octets. Aucune modalité, aucun schéma, aucune sémantique n'est jamais déclarée.
2. Le réseau n'agit que par ``read`` (observer) et ``intervene`` (poignées abstraites dont le sens
   n'est jamais révélé). Une intervention crée une *branche* (id >= 1, croissant) ; la branche 0 est
   le flux observationnel. Les branches persistent pendant tout l'épisode.
3. ``read`` est idempotent : même requête => mêmes octets (le bruit est fixé à la création de la
   branche ou de la vue). Lecture partiellement hors flux : tronquée ; entièrement hors flux : b"".
4. Une *vue* est un rendu alternatif du même état latent (autre encodage, bruit, occultation).
   En entraînement, les vues d'un épisode sont des positifs contrastifs garantis ; en production
   n_views = 1 (le réseau fabrique alors ses propres transformations génériques).
5. L'agent ne reçoit jamais le juge : ``Episode`` = ``EpisodeEnv`` (agent) + ``EpisodeJudge``
   (entraîneur). Le runtime n'expose à l'agent que l'enveloppe ``Session`` (Partie 5).
6. Les slots d'unités de ``Understanding`` sont non ordonnés : l'appariement avec la vérité cachée
   est la responsabilité du juge.
7. Exigences sur ``judge`` (alignement des incitations avec le critère de réussite) :
   - justesse    : seules les affirmations comptent ; une affirmation fausse coûte plus qu'une abstention ;
   - couverture  : part de la vérité *identifiable avec le budget accordé* effectivement comprise ;
   - calibration : 1 - erreur d'étalonnage (confiances, identifiabilité, abstentions) ;
   - aucune récompense pour l'invention d'informations non identifiables ;
   - le juge accepte une compréhension vide et des réponses d'abstention (cf. check_world_contract).
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

API_VERSION = "1.0"
SPLITS: Tuple[str, ...] = ("train", "val", "ood")
N_CAUSAL_CLASSES = 3


class CausalClass(enum.IntEnum):
    NONE = 0         # aucune dépendance
    ASSOCIATIVE = 1  # association sans effet interventionnel direct (confusion, médiation…)
    CAUSAL = 2       # dépendance causale directe i -> j


class QueryKind(enum.IntEnum):
    CHOICE = 0       # choisir parmi des candidats encodés en octets
    SCALAR = 1       # valeur réelle (moyenne + écart-type)
    BYTES = 2        # séquence d'octets


# --------------------------------------------------------------------------- erreurs

class ApiError(Exception):
    """Racine des erreurs du contrat."""


class InvalidRequest(ApiError):
    """Requête illégale de l'agent (branche inconnue, vue inexistante, intervention impossible…)."""


class BudgetExceeded(InvalidRequest):
    """Budget d'épisode dépassé."""


class WorldContractError(ApiError):
    """Le monde viole le contrat (spec invalide, réponse mal formée…)."""


class MalformedOutput(ApiError):
    """La sortie du réseau (Understanding, Answer) est mal formée."""


# --------------------------------------------------------------------------- validations

def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _check_array(name: str, arr: Any, shape: Tuple[Optional[int], ...], *, kind: str = "f",
                 exc: type = WorldContractError) -> np.ndarray:
    """kind : 'f' flottant, 'i' entier, 'b' booléen. ``None`` dans shape = dimension libre."""
    if not isinstance(arr, np.ndarray):
        raise exc(f"{name} : np.ndarray attendu, reçu {type(arr).__name__}")
    if arr.ndim != len(shape) or any(s is not None and a != s for a, s in zip(arr.shape, shape)):
        raise exc(f"{name} : forme {arr.shape} incompatible avec {shape}")
    if kind == "b":
        good = arr.dtype == np.bool_
    elif kind == "i":
        good = np.issubdtype(arr.dtype, np.integer)
    else:
        good = np.issubdtype(arr.dtype, np.floating)
    if not good:
        raise exc(f"{name} : dtype {arr.dtype} incompatible (attendu '{kind}')")
    if kind == "f" and not bool(np.all(np.isfinite(arr))):
        raise exc(f"{name} : valeurs non finies")
    return arr


def _check_prob(name: str, arr: np.ndarray, exc: type, tol: float = 1e-4) -> None:
    if arr.size and (float(arr.min()) < -tol or float(arr.max()) > 1.0 + tol):
        raise exc(f"{name} : valeurs hors de [0, 1]")


# --------------------------------------------------------------------------- spécification d'épisode

@dataclass(frozen=True)
class Budget:
    max_steps: int = 24
    max_reads: int = 16
    max_bytes: int = 1 << 16
    max_interventions: int = 8

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise WorldContractError("Budget.max_steps doit être >= 1")
        for name in ("max_reads", "max_bytes", "max_interventions"):
            if getattr(self, name) < 0:
                raise WorldContractError(f"Budget.{name} doit être >= 0")


@dataclass(frozen=True)
class EpisodeSpec:
    """Tout ce que l'agent a le droit de savoir avant d'agir. Aucune sémantique."""

    episode_id: str                              # journalisation uniquement ; le réseau ne l'utilise pas
    stream_length: int                           # longueur (octets) de la branche 0, vue 0
    n_views: int = 1
    n_handles: int = 0                           # 0 = épisode purement observationnel
    handle_dim: int = 0                          # dimension du vecteur de valeur d'une poignée
    handle_value_range: Tuple[float, float] = (-1.0, 1.0)
    shared_exogenous_noise: bool = False         # une branche partage-t-elle le bruit exogène de sa base ?
    budget: Budget = field(default_factory=Budget)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise WorldContractError("EpisodeSpec.episode_id vide")
        if self.stream_length < 1:
            raise WorldContractError("EpisodeSpec.stream_length doit être >= 1")
        if self.n_views < 1:
            raise WorldContractError("EpisodeSpec.n_views doit être >= 1")
        if self.n_handles < 0 or self.handle_dim < 0:
            raise WorldContractError("n_handles et handle_dim doivent être >= 0")
        if self.n_handles > 0 and self.handle_dim < 1:
            raise WorldContractError("handle_dim >= 1 requis quand n_handles > 0")
        lo, hi = self.handle_value_range
        if not (_finite(lo) and _finite(hi) and lo < hi):
            raise WorldContractError("handle_value_range invalide")


# --------------------------------------------------------------------------- actions de l'agent

@dataclass(frozen=True)
class ReadRequest:
    branch: int = 0
    offset: int = 0
    length: int = 1024
    view: int = 0

    def __post_init__(self) -> None:
        if self.branch < 0 or self.offset < 0 or self.view < 0 or self.length < 1:
            raise InvalidRequest(f"ReadRequest invalide : {self!r}")


@dataclass(frozen=True)
class InterventionRequest:
    """do(poignée = valeur) appliqué à ``base_branch`` ; crée une nouvelle branche."""

    handle: int
    value: Tuple[float, ...] = ()
    base_branch: int = 0

    def __post_init__(self) -> None:
        if self.handle < 0 or self.base_branch < 0:
            raise InvalidRequest(f"InterventionRequest invalide : {self!r}")
        object.__setattr__(self, "value", tuple(float(v) for v in self.value))
        if not all(math.isfinite(v) for v in self.value):
            raise InvalidRequest("InterventionRequest.value contient des valeurs non finies")

    def as_array(self) -> np.ndarray:
        return np.asarray(self.value, dtype=np.float32)


# --------------------------------------------------------------------------- requêtes et réponses

@dataclass(frozen=True)
class Query:
    """Question inédite posée en fin d'épisode. Le prompt et les candidats sont des octets bruts."""

    query_id: str
    kind: QueryKind
    prompt: bytes
    candidates: Tuple[bytes, ...] = ()
    scalar_range: Optional[Tuple[float, float]] = None
    max_answer_bytes: int = 256
    branch: int = 0                                    # branche sur laquelle porte la question
    hypothetical: Optional[InterventionRequest] = None  # modification hypothétique, NON exécutée pour l'agent

    def __post_init__(self) -> None:
        if not self.query_id:
            raise WorldContractError("Query.query_id vide")
        if not isinstance(self.prompt, (bytes, bytearray)):
            raise WorldContractError("Query.prompt doit être de type bytes")
        if any(not isinstance(c, (bytes, bytearray)) for c in self.candidates):
            raise WorldContractError("Query.candidates doit contenir des bytes")
        if self.kind == QueryKind.CHOICE and len(self.candidates) < 2:
            raise WorldContractError("Query CHOICE : au moins 2 candidats")
        if self.kind == QueryKind.SCALAR and (self.scalar_range is None or
                                              not self.scalar_range[0] < self.scalar_range[1]):
            raise WorldContractError("Query SCALAR : scalar_range=(lo, hi) avec lo < hi requis")
        if self.kind == QueryKind.BYTES and self.max_answer_bytes < 1:
            raise WorldContractError("Query BYTES : max_answer_bytes >= 1")


@dataclass(eq=False)
class Answer:
    query_id: str
    kind: QueryKind
    choice_probs: Optional[np.ndarray] = None   # [n_candidats], somme 1
    scalar_mean: Optional[float] = None
    scalar_std: Optional[float] = None
    payload: Optional[bytes] = None
    confidence: float = 0.0                     # P(la réponse est correcte), calibrée
    abstain_prob: float = 0.0                   # P(la réponse n'est pas déterminable)

    @classmethod
    def abstain(cls, q: Query) -> "Answer":
        if q.kind == QueryKind.CHOICE:
            n = len(q.candidates)
            return cls(q.query_id, q.kind, choice_probs=np.full(n, 1.0 / n, np.float32),
                       confidence=0.0, abstain_prob=1.0)
        if q.kind == QueryKind.SCALAR:
            lo, hi = q.scalar_range  # type: ignore[misc]
            return cls(q.query_id, q.kind, scalar_mean=0.5 * (lo + hi), scalar_std=float(hi - lo),
                       confidence=0.0, abstain_prob=1.0)
        return cls(q.query_id, q.kind, payload=b"", confidence=0.0, abstain_prob=1.0)

    def validate(self, q: Query) -> None:
        if self.query_id != q.query_id or self.kind != q.kind:
            raise MalformedOutput("Answer ne correspond pas à la Query")
        if not (_finite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise MalformedOutput("Answer.confidence hors de [0, 1]")
        if not (_finite(self.abstain_prob) and 0.0 <= self.abstain_prob <= 1.0):
            raise MalformedOutput("Answer.abstain_prob hors de [0, 1]")
        if q.kind == QueryKind.CHOICE:
            p = _check_array("choice_probs", self.choice_probs, (len(q.candidates),), exc=MalformedOutput)
            if float(p.min()) < -1e-6 or abs(float(p.sum()) - 1.0) > 1e-3:
                raise MalformedOutput("choice_probs : distribution invalide")
        elif q.kind == QueryKind.SCALAR:
            if not (_finite(self.scalar_mean) and _finite(self.scalar_std) and float(self.scalar_std) >= 0.0):
                raise MalformedOutput("scalar_mean/scalar_std invalides")
        else:
            if not isinstance(self.payload, (bytes, bytearray)) or len(self.payload) > q.max_answer_bytes:
                raise MalformedOutput("payload invalide ou trop long")


# --------------------------------------------------------------------------- compréhension (sortie du réseau)

@dataclass(eq=False)
class Understanding:
    """
    Compréhension construite pour un épisode. Tout est NumPy float32 sauf ``patch_spans`` (int64).

    patch_spans            [P, 4]   (branche, vue, début, fin exclue) : repère des appartenances
    unit_presence          [M]      P(le slot désigne une unité réelle)
    unit_membership        [M, P]   appartenance souple de chaque patch à chaque unité
    unit_embedding         [M, Du]
    dependency             [M, M]   P(i influence j) ; diagonale nulle ; orienté, non symétrique
    causal_class           [M, M, 3] probabilités (NONE, ASSOCIATIVE, CAUSAL) ; somme 1
    condition_gate         [M, M, M] P(l'unité k conditionne la relation i -> j)
    relation_confidence    [M, M]   P(l'estimation de la relation est correcte), calibrée
    relation_identifiable  [M, M]   P(la relation est identifiable avec les preuves récoltées)
    mechanism_embedding    [M, Dm]
    unexplained            [P]      P(le patch reste inexpliqué)
    handle_effect          [H, M]   effet mesuré de chaque poignée sur chaque unité
    coverage_claim         fraction de l'entrée que le réseau affirme avoir expliquée
    """

    episode_id: str
    patch_spans: np.ndarray
    unit_presence: np.ndarray
    unit_membership: np.ndarray
    unit_embedding: np.ndarray
    dependency: np.ndarray
    causal_class: np.ndarray
    condition_gate: np.ndarray
    relation_confidence: np.ndarray
    relation_identifiable: np.ndarray
    mechanism_embedding: np.ndarray
    unexplained: np.ndarray
    handle_effect: np.ndarray
    coverage_claim: float
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.unit_presence.shape[0])

    @property
    def n_patches(self) -> int:
        return int(self.patch_spans.shape[0])

    def validate(self) -> None:
        exc = MalformedOutput
        presence = _check_array("unit_presence", self.unit_presence, (None,), exc=exc)
        m = presence.shape[0]
        spans = _check_array("patch_spans", self.patch_spans, (None, 4), kind="i", exc=exc)
        p = spans.shape[0]
        if p and bool(np.any(spans[:, 3] < spans[:, 2])):
            raise exc("patch_spans : fin < début")
        membership = _check_array("unit_membership", self.unit_membership, (m, p), exc=exc)
        _check_array("unit_embedding", self.unit_embedding, (m, None), exc=exc)
        dep = _check_array("dependency", self.dependency, (m, m), exc=exc)
        causal = _check_array("causal_class", self.causal_class, (m, m, N_CAUSAL_CLASSES), exc=exc)
        gate = _check_array("condition_gate", self.condition_gate, (m, m, m), exc=exc)
        conf = _check_array("relation_confidence", self.relation_confidence, (m, m), exc=exc)
        ident = _check_array("relation_identifiable", self.relation_identifiable, (m, m), exc=exc)
        _check_array("mechanism_embedding", self.mechanism_embedding, (m, None), exc=exc)
        unexplained = _check_array("unexplained", self.unexplained, (p,), exc=exc)
        _check_array("handle_effect", self.handle_effect, (None, m), exc=exc)
        for name, arr in (("unit_presence", presence), ("unit_membership", membership), ("dependency", dep),
                          ("causal_class", causal), ("condition_gate", gate), ("relation_confidence", conf),
                          ("relation_identifiable", ident), ("unexplained", unexplained)):
            _check_prob(name, arr, exc)
        if m and not np.allclose(np.diagonal(dep), 0.0, atol=1e-6):
            raise exc("dependency : diagonale non nulle")
        if m and not np.allclose(causal.sum(-1), 1.0, atol=1e-3):
            raise exc("causal_class : les probabilités ne somment pas à 1")
        if not (_finite(self.coverage_claim) and 0.0 <= self.coverage_claim <= 1.0):
            raise exc("coverage_claim hors de [0, 1]")

    @classmethod
    def blank(cls, episode_id: str, *, n_units: int = 4, n_patches: int = 2, d_unit: int = 8,
              d_mech: int = 8, n_handles: int = 0) -> "Understanding":
        """Compréhension vide (aucune affirmation) : sert à tester le juge."""
        m, p = n_units, n_patches
        causal = np.zeros((m, m, N_CAUSAL_CLASSES), np.float32)
        causal[..., CausalClass.NONE] = 1.0
        spans = np.zeros((p, 4), np.int64)
        spans[:, 2] = np.arange(p)
        spans[:, 3] = np.arange(p) + 1
        return cls(
            episode_id=episode_id,
            patch_spans=spans,
            unit_presence=np.zeros(m, np.float32),
            unit_membership=np.zeros((m, p), np.float32),
            unit_embedding=np.zeros((m, d_unit), np.float32),
            dependency=np.zeros((m, m), np.float32),
            causal_class=causal,
            condition_gate=np.zeros((m, m, m), np.float32),
            relation_confidence=np.zeros((m, m), np.float32),
            relation_identifiable=np.zeros((m, m), np.float32),
            mechanism_embedding=np.zeros((m, d_mech), np.float32),
            unexplained=np.ones(p, np.float32),
            handle_effect=np.zeros((n_handles, m), np.float32),
            coverage_claim=0.0,
        )


# --------------------------------------------------------------------------- verdict (sortie du juge)

@dataclass(frozen=True)
class Score:
    """Les trois critères du projet, tous dans [0, 1], 1 = meilleur."""

    justesse: float
    couverture: float
    calibration: float
    details: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("justesse", "couverture", "calibration"):
            v = getattr(self, name)
            if not (_finite(v) and -1e-6 <= float(v) <= 1.0 + 1e-6):
                raise WorldContractError(f"Score.{name} doit être dans [0, 1], reçu {v!r}")

    def total(self, w_justesse: float = 1.0, w_couverture: float = 1.0, w_calibration: float = 1.0) -> float:
        z = w_justesse + w_couverture + w_calibration
        if z <= 0:
            raise ValueError("somme des poids nulle")
        return (w_justesse * self.justesse + w_couverture * self.couverture
                + w_calibration * self.calibration) / z


@dataclass(eq=False)
class Judgement:
    """
    Retour du juge, dans le repère des slots du réseau (le juge fait l'appariement).

    unit_quality           [M]    qualité d'appariement du slot (0 si fantôme)
    unit_matched           [M]    bool : slot apparié à une unité réelle
    relation_correct       [M,M]  correction de la classe causale annoncée (définie où relation_mask)
    relation_mask          [M,M]  bool : entrées évaluables
    relation_identifiable  [M,M]  bool : la vérité est-elle identifiable avec les preuves disponibles ?
    answer_correct         [Q]    correction de chaque réponse
    answer_identifiable    [Q]    bool
    """

    score: Score
    unit_quality: np.ndarray
    unit_matched: np.ndarray
    relation_correct: np.ndarray
    relation_mask: np.ndarray
    relation_identifiable: np.ndarray
    answer_correct: np.ndarray
    answer_identifiable: np.ndarray
    info: Dict[str, float] = field(default_factory=dict)

    def validate(self, n_units: int, n_queries: int) -> None:
        exc = WorldContractError
        if not isinstance(self.score, Score):
            raise exc("Judgement.score doit être un Score")
        for name, shape in (("unit_quality", (n_units,)), ("relation_correct", (n_units, n_units)),
                            ("answer_correct", (n_queries,))):
            _check_prob(name, _check_array(name, getattr(self, name), shape, exc=exc), exc)
        for name, shape in (("unit_matched", (n_units,)), ("relation_mask", (n_units, n_units)),
                            ("relation_identifiable", (n_units, n_units)),
                            ("answer_identifiable", (n_queries,))):
            _check_array(name, getattr(self, name), shape, kind="b", exc=exc)


# --------------------------------------------------------------------------- protocoles à implémenter côté monde

@runtime_checkable
class EpisodeEnv(Protocol):
    """Ce que l'agent peut appeler (via Session)."""

    @property
    def spec(self) -> EpisodeSpec: ...

    def branch_length(self, branch: int, view: int = 0) -> int:
        """Longueur en octets du flux (branche, vue). Branche inconnue : InvalidRequest."""

    def read(self, req: ReadRequest) -> bytes:
        """Octets [offset, offset+length) du flux (branche, vue). Idempotent. Hors budget : le runtime filtre."""

    def intervene(self, req: InterventionRequest) -> int:
        """Applique do(poignée=valeur) à req.base_branch ; renvoie l'id (>= 1) de la nouvelle branche."""

    def queries(self) -> Sequence[Query]:
        """Requêtes inédites posées en fin d'épisode (peut être vide)."""


@runtime_checkable
class EpisodeJudge(Protocol):
    """Ce que seul l'entraîneur appelle. Jamais exposé à l'agent."""

    @property
    def meta(self) -> Mapping[str, Any]:
        """Métadonnées de journalisation (famille, difficulté…). Opaques pour le réseau."""

    def judge(self, understanding: Understanding, answers: Sequence[Answer]) -> Judgement: ...


@runtime_checkable
class Episode(EpisodeEnv, EpisodeJudge, Protocol):
    """Un épisode complet côté monde = vue de l'agent + vue de l'entraîneur."""


@runtime_checkable
class World(Protocol):
    def families(self, split: str) -> Sequence[str]:
        """Familles de mécanismes/transformations/encodages disponibles ; 'ood' disjoint de 'train'."""

    def sample(self, *, split: str, seed: int, family: Optional[str] = None) -> Episode:
        """Déterministe en (split, seed, family)."""


# --------------------------------------------------------------------------- test de conformité

def check_world_contract(world: World, *, split: str = "train", seed: int = 0,
                         family: Optional[str] = None) -> Dict[str, Any]:
    """
    À lancer sur votre générateur avant tout entraînement.
    Lève WorldContractError à la première violation ; sinon renvoie un petit rapport.
    """
    if split not in SPLITS:
        raise WorldContractError(f"split inconnu : {split!r}")
    if not isinstance(world, World):
        raise WorldContractError("l'objet ne respecte pas le protocole World")
    ep = world.sample(split=split, seed=seed, family=family)
    if not isinstance(ep, Episode):
        raise WorldContractError("sample() doit renvoyer un Episode (EpisodeEnv + EpisodeJudge)")
    spec = ep.spec
    if not isinstance(spec, EpisodeSpec):
        raise WorldContractError("Episode.spec doit être un EpisodeSpec")

    n0 = ep.branch_length(0, 0)
    if not (1 <= n0 <= spec.stream_length):
        raise WorldContractError(f"branch_length(0) = {n0} hors de [1, {spec.stream_length}]")
    length = min(512, n0)
    req = ReadRequest(0, 0, length, 0)
    ref = ep.read(req)
    if not isinstance(ref, (bytes, bytearray)) or not (1 <= len(ref) <= length):
        raise WorldContractError("read() doit renvoyer entre 1 et `length` octets pour une lecture dans le flux")
    if bytes(ep.read(req)) != bytes(ref):
        raise WorldContractError("read() n'est pas idempotent")
    twin = world.sample(split=split, seed=seed, family=family)
    if bytes(twin.read(req)) != bytes(ref):
        raise WorldContractError("monde non déterministe pour (split, seed, family)")
    for v in range(spec.n_views):
        if not isinstance(ep.read(ReadRequest(0, 0, length, v)), (bytes, bytearray)):
            raise WorldContractError(f"read(view={v}) doit renvoyer des octets")
    if bytes(ep.read(ReadRequest(0, n0, 16, 0))) != b"":
        raise WorldContractError("une lecture entièrement hors flux doit renvoyer b''")

    n_branches = 1
    if spec.n_handles > 0 and spec.budget.max_interventions > 0:
        lo, hi = spec.handle_value_range
        mid = tuple(0.5 * (lo + hi) for _ in range(spec.handle_dim))
        branch = ep.intervene(InterventionRequest(0, mid, 0))
        if not isinstance(branch, int) or branch < 1:
            raise WorldContractError("intervene() doit renvoyer un id de branche entier >= 1")
        nb = ep.branch_length(branch, 0)
        if nb < 1:
            raise WorldContractError("la branche d'intervention est vide")
        if not isinstance(ep.read(ReadRequest(branch, 0, min(length, nb), 0)), (bytes, bytearray)):
            raise WorldContractError("read() sur une branche d'intervention doit renvoyer des octets")
        n_branches = 2

    queries = tuple(ep.queries())
    for q in queries:
        if not isinstance(q, Query):
            raise WorldContractError("queries() doit renvoyer des Query")
    answers = [Answer.abstain(q) for q in queries]
    blank = Understanding.blank(spec.episode_id, n_units=4, n_patches=2, n_handles=spec.n_handles)
    blank.validate()
    verdict = ep.judge(blank, answers)
    if not isinstance(verdict, Judgement):
        raise WorldContractError("judge() doit renvoyer un Judgement")
    verdict.validate(blank.n_units, len(queries))
    return {
        "family": ep.meta.get("family"),
        "stream_length": n0,
        "n_views": spec.n_views,
        "n_handles": spec.n_handles,
        "n_branches_tested": n_branches,
        "n_queries": len(queries),
        "blank_total_score": verdict.score.total(),
    }