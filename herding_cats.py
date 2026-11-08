"""
herding_cats.py — toolkit for the "Herding Quantum Cats" hackathon
Qiskit Fall Fest 2026 · University of Stuttgart

You do NOT need to read this file to take part, but you are welcome to: every
function is short and documented, and nothing in here is magic. Advanced teams
are encouraged to copy, modify or replace any of it.

Tested with: qiskit 2.6.1, qiskit-ibm-runtime 0.51.0, qiskit-aer 0.18.0

Contents
--------
1. Backends ............ fake_kingston(), fake_yorktown(), connect_to_hardware()
2. Cat circuits ........ ghz_linear(), baseline_cat(), cat_qubits()
3. Measuring a cat ..... measure_cat(), cat_fidelity(), CatResult
4. Circuit diagnostics . cat_report(), schedule_circuit()
5. Simulator ........... CatSimulator  (the "Fall Fest noise model")
6. Mitigation pipeline . Recipe, build_cat_experiment(), measure_cat_quality(), fold_cz()
7. Hardware ............ HardwareRunner, retrieve_job_counts()
8. Verification ........ verify_cat()
9. Submission & more ... save_submission(), FinalJob, scorecard(), show_chip(),
                         qubit_quality(), save_circuit()/load_circuit()
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qpy
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate
from qiskit.quantum_info import Clifford, Pauli, StabilizerState
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDelay,
    PadDynamicalDecoupling,
)

__version__ = "1.0.0"

#: Size of the cat for the "Cleanest Cat" leaderboard.
CLEAN_CAT_SIZE = 12
#: Shot budget the organizers' scoring run gives every team (split over all circuits).
SCORING_SHOTS = 100_000
#: Default shot budget for your own quick estimates with scorecard().
ESTIMATE_SHOTS = 24_000
#: Seed that defines the organizers' baseline cat. Do not change.
BASELINE_SEED = 2016


# =============================================================================
# 1. Backends
# =============================================================================
def fake_kingston():
    """A snapshot of ibm_kingston (156-qubit IBM Heron r2, calibration of April 2026).

    ibm_kingston is available to every IBM Quantum Open Plan user, so this is the
    machine you develop against today and run on at the end.
    """
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    return FakeKingston()


def fake_yorktown():
    """A snapshot of ibmqx2 "Yorktown": one of the first 5-qubit processors on the
    IBM Quantum Experience (online 2017-2021). Ten years of progress in one object.
    """
    from qiskit_ibm_runtime.fake_provider import FakeYorktownV2

    return FakeYorktownV2()


def connect_to_hardware(backend_name: str = "ibm_kingston", **service_kwargs):
    """Return a real IBM Quantum backend.

    Requires that you saved your account beforehand, e.g.::

        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(token="<API key>", instance="<CRN>",
                                          set_as_default=True)
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(**service_kwargs)
    return service.backend(backend_name)


# =============================================================================
# 2. Cat circuits
# =============================================================================
def ghz_linear(n: int) -> QuantumCircuit:
    """The textbook cat: a Hadamard followed by a chain of CNOTs (depth grows like n)."""
    qc = QuantumCircuit(n, name=f"ghz_linear_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def baseline_cat(backend, n: int = CLEAN_CAT_SIZE) -> QuantumCircuit:
    """The organizers' reference cat: ghz_linear(n), transpiled with the default
    preset pass manager (optimization_level=1) and a fixed seed.

    Your improvement is always measured against THIS circuit, run in the same job.
    """
    pass_manager = generate_preset_pass_manager(
        optimization_level=1, backend=backend, seed_transpiler=BASELINE_SEED
    )
    return pass_manager.run(ghz_linear(n))


def cat_qubits(circuit: QuantumCircuit) -> list[int]:
    """Physical qubits that hold the cat, in the order of the original (virtual) qubits.

    Works for any circuit produced by a Qiskit pass manager; accounts for the
    final permutation introduced by SWAP routing.
    """
    if circuit.layout is None:
        raise ValueError(
            "This circuit has no layout information. Either transpile it with a "
            "pass manager, or pass the physical qubits explicitly with qubits=[...]."
        )
    return list(circuit.layout.final_index_layout(filter_ancillas=True))


# =============================================================================
# 3. Measuring a cat
# =============================================================================
def measure_cat(
    prep: QuantumCircuit,
    basis: str = "Z",
    qubits: Sequence[int] | None = None,
    flip_mask: Sequence[int] | None = None,
) -> QuantumCircuit:
    """Return a copy of `prep` that measures the cat in the Z or X basis.

    * Bit i of the classical register "meas" holds cat qubit i.
    * The X-basis rotation is written in native gates (RZ·SX·RZ = H up to phase),
      so the result is still a valid ISA circuit for an IBM Heron backend.
    * `flip_mask` (used for readout twirling) adds an X gate right before the
      measurement of every cat qubit whose mask entry is 1.
    """
    qubits = list(qubits) if qubits is not None else cat_qubits(prep)
    qc = prep.copy()
    qc.name = f"{prep.name}_{basis}"
    basis = basis.upper()
    if basis not in ("Z", "X"):
        raise ValueError("basis must be 'Z' or 'X'")
    if basis == "X":
        for qubit in qubits:
            qc.rz(np.pi / 2, qubit)
            qc.sx(qubit)
            qc.rz(np.pi / 2, qubit)
    if flip_mask is not None:
        for qubit, flip in zip(qubits, flip_mask):
            if flip:
                qc.x(qubit)
    creg = ClassicalRegister(len(qubits), "meas")
    qc.add_register(creg)
    for position, qubit in enumerate(qubits):
        qc.measure(qubit, creg[position])
    return qc


def _counts_to_arrays(counts: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a counts dict into (bits[m, n], weights[m]); bits[:, i] is cat qubit i.

    Each of the m rows is one distinct measured bitstring; weights holds how many
    shots gave it.
    """
    bitstrings = list(counts.keys())
    weights = np.array([counts[key] for key in bitstrings], dtype=float)
    bits = np.zeros((len(bitstrings), n), dtype=np.int8)
    for row, bitstring in enumerate(bitstrings):
        bitstring = bitstring.replace(" ", "")
        if bitstring.startswith("0x"):
            bitstring = format(int(bitstring, 16), f"0{n}b")
        bitstring = bitstring[-n:].zfill(n)
        # Qiskit bitstrings are little-endian: reverse so that column i is cat qubit i
        bits[row] = np.frombuffer(bitstring[::-1].encode(), dtype=np.uint8) - 48
    return bits, weights


@dataclass
class CatResult:
    """Fidelity estimate of a GHZ ("cat") state.

    F  = P + (X - 1)/2   is a rigorous LOWER bound on the GHZ fidelity, where
    P  = probability that all qubits agree (all 0 or all 1)  -> "is the cat alive OR dead?"
    X  = <X X ... X>, the parity in the X basis               -> "is it in superposition?"

    A perfect cat has F = 1. A classical coin (50% all-0, 50% all-1) has P = 1,
    X = 0 and therefore F = 0.5: you must beat 0.5 to prove real N-qubit entanglement.
    """

    F: float
    F_err: float
    P: float
    P_err: float
    X: float
    X_err: float
    n: int
    shots: int = 0
    details: dict = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        """True if F is above 0.5 by at least two standard errors (genuine N-qubit entanglement)."""
        return self.F - 2 * self.F_err > 0.5

    @property
    def error(self) -> float:
        """Distance to a perfect cat, |1 - F|."""
        return abs(1.0 - self.F)

    def __repr__(self) -> str:
        tag = "certified ✓" if self.certified else "not certified"
        return (
            f"CatResult(N={self.n}: F = {self.F:.3f} ± {self.F_err:.3f} [{tag}] | "
            f"P = {self.P:.3f} ± {self.P_err:.3f}, X = {self.X:.3f} ± {self.X_err:.3f})"
        )


def cat_fidelity(counts_z: dict, counts_x: dict, n: int | None = None) -> CatResult:
    """Raw (unmitigated) fidelity bound from one Z-basis and one X-basis measurement."""
    if n is None:
        n = len(next(iter(counts_z)).replace(" ", ""))
    bits_z, weights_z = _counts_to_arrays(counts_z, n)
    bits_x, weights_x = _counts_to_arrays(counts_x, n)
    shots_z, shots_x = weights_z.sum(), weights_x.sum()
    # P: fraction of Z-basis shots in which all qubits agree
    all_equal = (bits_z.min(axis=1) == bits_z.max(axis=1)).astype(float)
    P = float((all_equal * weights_z).sum() / shots_z)
    # X: average of (-1)^(number of ones) over the X-basis shots
    parity = 1.0 - 2.0 * (bits_x.sum(axis=1) % 2)
    X = float((parity * weights_x).sum() / shots_x)
    # statistical errors: binomial for P, standard error of a ±1 mean for X, combined in quadrature
    P_err = math.sqrt(max(P * (1 - P), 1e-12) / shots_z)
    X_err = math.sqrt(max(1 - X * X, 1e-12) / shots_x)
    F = P + (X - 1) / 2
    F_err = math.sqrt(P_err**2 + (X_err / 2) ** 2)
    return CatResult(F, F_err, P, P_err, X, X_err, n, int(shots_z + shots_x))


# =============================================================================
# 4. Circuit diagnostics
# =============================================================================
def schedule_circuit(circuit: QuantumCircuit, backend, dd: str | None = None) -> QuantumCircuit:
    """Insert explicit idle periods (delays), optionally filled with dynamical decoupling.

    dd=None : leave idle periods empty (they still decohere!)
    dd="XX" : fill every long-enough idle period with an X-X echo (a Hahn/CPMG echo,
              borrowed straight from NMR).
    """
    target = backend.target
    passes = [ALAPScheduleAnalysis(target=target)]
    if dd is not None:
        if dd.upper() != "XX":
            raise ValueError("Supported dd sequences: None or 'XX' (X is native on Heron).")
        passes.append(PadDynamicalDecoupling(target=target, dd_sequence=[XGate(), XGate()]))
    passes.append(PadDelay(target=target))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PassManager(passes).run(circuit)


def _two_qubit_ops(circuit: QuantumCircuit) -> list:
    return [i for i in circuit.data if i.operation.num_qubits == 2 and i.operation.name != "barrier"]


def cat_report(prep: QuantumCircuit, backend, qubits: Sequence[int] | None = None) -> dict:
    """Quick, simulation-free diagnostics of a cat-preparation circuit.

    Returns two-qubit gate count, two-qubit depth, duration, total idle time of
    the cat qubits *after* they joined the cat (when they are fragile), and an
    "estimated success probability" (ESP) built from the backend's calibration data:

        ESP = Π(1 - gate errors) × Π(1 - readout errors) × Π exp(-idle time / T2)

    ESP is a crude but very fast proxy for cat quality — handy for comparing many
    candidate layouts without running a simulation.
    """
    qubits = list(qubits) if qubits is not None else cat_qubits(prep)
    target = backend.target
    sched = schedule_circuit(prep, backend)
    dt = backend.dt
    log_esp = 0.0
    n2q = 0
    for inst in prep.data:
        op = inst.operation
        if op.name in ("barrier", "delay", "measure"):
            continue
        qs = tuple(prep.find_bit(q).index for q in inst.qubits)
        if op.num_qubits == 2:
            n2q += 1
        try:
            err = target[op.name][qs].error or 0.0
        except Exception:
            err = 0.0
        log_esp += math.log(max(1e-12, 1 - err))
    for q in qubits:
        try:
            err = target["measure"][(q,)].error or 0.0
        except Exception:
            err = 0.0
        log_esp += math.log(max(1e-12, 1 - err))
    # idle time after the qubit has been entangled (first 2-qubit gate on it)
    entangled = set()
    idle_after = {q: 0.0 for q in qubits}
    for inst in sched.data:
        qs = [sched.find_bit(q).index for q in inst.qubits]
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
            entangled.update(qs)
        elif inst.operation.name == "delay":
            q = qs[0]
            if q in entangled and q in idle_after:
                idle_after[q] += inst.operation.duration * (dt if inst.operation.unit == "dt" else 1.0)
    for q, t in idle_after.items():
        t2 = getattr(target.qubit_properties[q], "t2", None) if target.qubit_properties else None
        if t2:
            log_esp -= t / t2
    two_q_depth = prep.depth(lambda i: i.operation.num_qubits == 2 and i.operation.name != "barrier")
    duration = sched.duration * (dt if sched.unit == "dt" else 1.0) if sched.duration else float("nan")
    return {
        "n": len(qubits),
        "two_qubit_gates": n2q,
        "swap_overhead": max(0, n2q - (len(qubits) - 1)),
        "two_qubit_depth": two_q_depth,
        "duration_us": duration * 1e6,
        "idle_after_entangling_us": sum(idle_after.values()) * 1e6,
        "ESP": math.exp(log_esp),
        "qubits": qubits,
    }


# =============================================================================
# 5. Simulator: the "Fall Fest noise model"
# =============================================================================
def _active_qubits(circuits: Iterable[QuantumCircuit]) -> list[int]:
    act = set()
    for c in circuits:
        for inst in c.data:
            if inst.operation.name in ("delay", "barrier"):
                continue
            for q in inst.qubits:
                act.add(c.find_bit(q).index)
    return sorted(act)


def _chain_order(circuit: QuantumCircuit) -> list[int]:
    """Active qubits in depth-first order along the 2-qubit gates.

    Neighbours in the cat end up next to each other in the matrix-product state,
    which keeps Aer's internal qubit moves short (same result, faster)."""
    act = _active_qubits([circuit])
    adj: dict = {q: [] for q in act}
    for inst in circuit.data:
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
            a, b = (circuit.find_bit(q).index for q in inst.qubits)
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
    seen: set = set()
    order: list[int] = []
    starts = [q for q in act if adj[q]] + act
    for s in starts:
        stack = [s]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            order.append(v)
            stack.extend(w for w in reversed(adj[v]) if w not in seen)
    return order


def _pauli_twirled(error):
    """Pauli-twirl approximation of a QuantumError (same process fidelity).

    Aer samples Pauli channels much faster than general Kraus maps; for gate
    errors of ~0.1–1 % the difference to the full channel is negligible."""
    from qiskit.quantum_info import Kraus, pauli_basis
    from qiskit_aer.noise import pauli_error

    kraus = Kraus(error.to_quantumchannel()).data
    d = 2**error.num_qubits
    probs = []
    for P in pauli_basis(error.num_qubits):
        m = P.to_matrix().conj().T
        probs.append((P, float(sum(abs(np.trace(m @ k)) ** 2 for k in kraus)) / d**2))
    total = sum(p for _, p in probs)
    return pauli_error([(P, p / total) for P, p in probs if p > 1e-12])


class CatSimulator:
    """Noisy simulator of an IBM backend, tuned for cat states.

    Noise model = the backend's own calibration data (gate errors incl. T1/T2
    during gates, as Pauli channels; readout errors; full T1/T2 relaxation during
    idle periods) **plus** slow, random
    qubit-frequency noise during idle periods ("quasi-static detuning", standard
    deviation `detuning_khz`). The frequency noise is redrawn every
    shots/`noise_batches` shots. It is the kind of error a spin echo (dynamical
    decoupling) can undo — exactly as on the real hardware.

    Everything runs on Aer's matrix-product-state method, restricted to the qubits
    the circuit actually uses, so cats of 50+ qubits are fine.

    The organizers score the "Cleanest Cat" with this simulator, but with a secret
    `seed`: change `seed` yourself to check that your strategy is robust.
    """

    def __init__(
        self,
        backend=None,
        detuning_khz: float = 10.0,
        noise_batches: int = 8,
        seed: int | None = 1234,
        method: str = "matrix_product_state",
    ):
        from qiskit_aer.noise import NoiseModel

        self.backend = backend if backend is not None else fake_kingston()
        self.detuning_khz = detuning_khz
        self.noise_batches = max(1, int(noise_batches))
        self.method = method
        self.rng = np.random.default_rng(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._full_noise = NoiseModel.from_backend(self.backend)
        self._nm_cache: dict = {}
        self._twirl_cache: dict = {}

    # -- internals -----------------------------------------------------------
    def _compact_noise_model(self, phys: tuple[int, ...]):
        if phys in self._nm_cache:
            return self._nm_cache[phys]
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.noise.passes import RelaxationNoisePass

        mp = {p: i for i, p in enumerate(phys)}
        nm = self._full_noise
        new = NoiseModel(basis_gates=nm.basis_gates)
        for name, d in nm._local_quantum_errors.items():
            for qs, err in d.items():
                if all(q in mp for q in qs):
                    key = (name, tuple(qs))
                    if key not in self._twirl_cache:
                        self._twirl_cache[key] = _pauli_twirled(err)
                    new.add_quantum_error(self._twirl_cache[key], name, [mp[q] for q in qs], warnings=False)
        for qs, err in nm._local_readout_errors.items():
            if all(q in mp for q in qs):
                new.add_readout_error(err, [mp[q] for q in qs], warnings=False)
        for p in nm._custom_noise_passes:
            if isinstance(p, RelaxationNoisePass):
                new._custom_noise_passes.append(
                    RelaxationNoisePass(
                        t1s=[p._t1s[q] for q in phys],
                        t2s=[p._t2s[q] for q in phys],
                        dt=p._dt,
                        op_types=Delay,
                        excited_state_populations=[p._p1s[q] for q in phys],
                    )
                )
        self._nm_cache[phys] = new
        return new

    def _compact(self, circ: QuantumCircuit, mp: dict, detuning_hz: np.ndarray | None) -> QuantumCircuit:
        out = QuantumCircuit(len(mp), name=circ.name)
        for cr in circ.cregs:
            out.add_register(cr)
        dt = self.backend.dt
        for inst in circ.data:
            idx = [circ.find_bit(q).index for q in inst.qubits]
            name = inst.operation.name
            if name in ("delay", "barrier"):
                idx = [i for i in idx if i in mp]
                if not idx:
                    continue
                if name == "barrier":
                    out.barrier([mp[i] for i in idx])
                    continue
            elif any(i not in mp for i in idx):
                raise ValueError(f"operation {name} acts on an inactive qubit")
            out.append(
                inst.operation,
                [out.qubits[mp[i]] for i in idx],
                [out.clbits[circ.find_bit(c).index] for c in inst.clbits],
            )
            if name == "delay" and detuning_hz is not None:
                op = inst.operation
                t = op.duration * (dt if op.unit == "dt" else 1.0)
                q = mp[idx[0]]
                out.rz(2 * np.pi * detuning_hz[q] * t, q)
        return out

    # -- public --------------------------------------------------------------
    def run(self, circuits: QuantumCircuit | Sequence[QuantumCircuit], shots: int = 4000) -> list[dict]:
        """Simulate measured ISA circuits; returns one counts dict per circuit."""
        from qiskit_aer import AerSimulator

        single = isinstance(circuits, QuantumCircuit)
        circuits = [circuits] if single else list(circuits)
        shots = int(shots)
        # Every circuit is split into a few "noise batches" (fresh frequency noise each),
        # so that every measurement basis sees at least `noise_batches` noise draws.
        per_circuit = 1
        if self.detuning_khz > 0:
            per_circuit = max(1, math.ceil(self.noise_batches / max(1, len(circuits) // 3)))
            per_circuit = min(per_circuit, max(1, shots // 50))
        groups: dict = {}  # (phys, shots) -> list of (circuit index, compact circuit)
        for i, circ in enumerate(circuits):
            sched = schedule_circuit(circ, self.backend)  # adds idle periods; keeps existing DD
            phys = tuple(_chain_order(sched))
            mp = {p: j for j, p in enumerate(phys)}
            split = [shots // per_circuit + (1 if b < shots % per_circuit else 0) for b in range(per_circuit)]
            for s in split:
                if s == 0:
                    continue
                det = (
                    self.rng.normal(0.0, self.detuning_khz * 1e3, size=len(phys))
                    if self.detuning_khz > 0
                    else None
                )
                groups.setdefault((phys, s), []).append((i, self._compact(sched, mp, det)))
        out: list[dict] = [dict() for _ in circuits]
        for (phys, s), items in groups.items():
            sim = AerSimulator(method=self.method, noise_model=self._compact_noise_model(phys))
            res = sim.run([c for _, c in items], shots=s, seed_simulator=int(self.rng.integers(2**31))).result()
            for k, (i, _) in enumerate(items):
                for key, v in res.get_counts(k).items():
                    out[i][key] = out[i].get(key, 0) + v
        return out[0] if single else out


# =============================================================================
# 6. Mitigation pipeline
# =============================================================================
@dataclass
class Recipe:
    """How to measure your cat. Every option is optional; the default is "raw".

    readout_mitigation : TREX-style readout correction (random X before every
                         measurement + calibration circuits). Fixes most readout error.
    gate_twirling      : Pauli-twirl the CZ gates (turns coherent gate errors into
                         random ones; helps ZNE on real hardware).
    randomizations     : number of random twirl instances (used if any twirling is on).
    dd                 : None or "XX" — dynamical decoupling in idle periods.
    zne_factors        : noise-amplification factors for zero-noise extrapolation,
                         e.g. (1, 3, 5): each CZ is repeated 1, 3 or 5 times.
    zne_extrapolator   : "linear", "exponential" or "quadratic" (needs 3+ factors;
                         with exactly 3 factors this is Richardson extrapolation).
    """

    readout_mitigation: bool = False
    gate_twirling: bool = False
    randomizations: int = 16
    dd: str | None = None
    zne_factors: tuple = (1,)
    zne_extrapolator: str = "exponential"

    def __post_init__(self):
        self.zne_factors = tuple(int(f) for f in self.zne_factors)
        if any(f < 1 or f % 2 == 0 for f in self.zne_factors):
            raise ValueError("zne_factors must be odd integers >= 1, e.g. (1, 3, 5)")
        if self.zne_extrapolator not in ("linear", "exponential", "quadratic"):
            raise ValueError("zne_extrapolator must be 'linear', 'exponential' or 'quadratic'")
        if self.zne_extrapolator == "quadratic" and len(set(self.zne_factors)) < 3:
            raise ValueError("a quadratic extrapolation needs at least 3 different zne_factors, e.g. (1, 3, 5)")

    @property
    def num_randomizations(self) -> int:
        return max(1, int(self.randomizations)) if (self.readout_mitigation or self.gate_twirling) else 1


def fold_cz(circuit: QuantumCircuit, factor: int) -> QuantumCircuit:
    """Noise amplification for ZNE: replace every CZ by CZ^factor (factor odd, CZ·CZ = I)."""
    if factor == 1:
        return circuit.copy()
    folded = circuit.copy_empty_like()
    for instruction in circuit.data:
        repeats = factor if instruction.operation.name == "cz" else 1
        for _ in range(repeats):
            folded.append(instruction.operation, instruction.qubits, instruction.clbits)
    return folded


def _fit_extrapolate(x, y, yerr, kind: str) -> tuple[float, float]:
    """Weighted fit of y(x) and extrapolation to x = 0. Returns (value, std)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yerr = np.maximum(np.asarray(yerr, float), 1e-9)
    if len(x) == 1:
        return float(y[0]), float(yerr[0])
    # weighted least squares for y = intercept + slope * x; A is the design matrix, W the weights
    if kind == "exponential" and np.all(y > 0.02):
        # fit log(y) linearly, i.e. y = exp(intercept) * exp(slope * x)
        log_y, log_y_err = np.log(y), yerr / y
        A = np.vstack([np.ones_like(x), x]).T
        W = np.diag(1 / log_y_err**2)
        cov = np.linalg.inv(A.T @ W @ A)
        intercept, slope = cov @ A.T @ W @ log_y
        value_at_zero = float(np.exp(intercept))
        return value_at_zero, float(value_at_zero * math.sqrt(cov[0, 0]))
    if kind == "quadratic" and len(set(x.tolist())) >= 3:
        # y = intercept + slope * x + curvature * x^2
        A = np.vstack([np.ones_like(x), x, x**2]).T
    else:
        A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(1 / yerr**2)
    cov = np.linalg.inv(A.T @ W @ A)
    coefficients = cov @ A.T @ W @ y
    return float(coefficients[0]), float(math.sqrt(cov[0, 0]))


def _mitigated_estimates(bits_z, weights_z, bits_x, weights_x, lambdas):
    """TREX-style estimators. lambdas[i] = <Z_i> measured on |0> with readout twirling.

    Parity:  <X...X> = <measured parity> / Π lambda_i
    Agreement probability: P = Σ_shots 2^-N [Π(1 + s_i/lambda_i) + Π(1 - s_i/lambda_i)] / shots
    (this equals the sum over all even-weight Z-strings, each corrected by its own
    Π lambda_i, computed shot by shot). s_i = ±1 is the measured value of qubit i.
    Returns (P, X, per-shot values for P, per-shot values for X).
    """
    lambdas = np.clip(lambdas, 1e-3, None)
    values_z = 1.0 - 2.0 * bits_z               # bit 0 -> +1, bit 1 -> -1
    corrected = values_z / lambdas
    per_shot_P = np.prod((1 + corrected) / 2, axis=1) + np.prod((1 - corrected) / 2, axis=1)
    parity = 1.0 - 2.0 * (bits_x.sum(axis=1) % 2)
    per_shot_X = parity / np.prod(lambdas)
    P = float((per_shot_P * weights_z).sum() / weights_z.sum())
    X = float((per_shot_X * weights_x).sum() / weights_x.sum())
    return P, X, per_shot_P, per_shot_X


class CatExperiment:
    """All circuits needed to measure one cat with one Recipe, plus the analysis.

    Typical use::

        exp = build_cat_experiment(prep, backend, Recipe(readout_mitigation=True))
        counts = runner.run(exp.circuits, shots=exp.shots_per_circuit(8000))
        result = exp.analyze(counts)
    """

    def __init__(self, prep, backend, recipe: Recipe, qubits=None, seed: int = 0):
        from qiskit.circuit import pauli_twirl_2q_gates

        self.recipe = recipe
        self.backend = backend
        self.qubits = list(qubits) if qubits is not None else cat_qubits(prep)
        self.n = len(self.qubits)
        self.prep = prep
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        n_randomizations = recipe.num_randomizations
        self.circuits: list[QuantumCircuit] = []
        self.labels: list[dict] = []

        def add_circuit(circ, factor, rand_index, basis, mask):
            if recipe.dd:
                circ = schedule_circuit(circ, backend, dd=recipe.dd)
            circ.name = f"cat_f{factor}_r{rand_index}_{basis}"
            self.circuits.append(circ)
            self.labels.append({"factor": factor, "rand": rand_index, "basis": basis, "mask": mask})

        for factor in recipe.zne_factors:
            folded = fold_cz(prep, factor)
            for rand_index in range(n_randomizations):
                body = folded
                if recipe.gate_twirling:
                    body = pauli_twirl_2q_gates(
                        folded, seed=int(rng.integers(2**31)), target=backend.target
                    )
                for basis in ("Z", "X"):
                    # readout twirling: random X before measuring (mask), undone in software in _merge
                    mask = rng.integers(0, 2, self.n).tolist() if recipe.readout_mitigation else [0] * self.n
                    add_circuit(measure_cat(body, basis, self.qubits, mask), factor, rand_index, basis, mask)
        if recipe.readout_mitigation:
            # readout calibration: measure |00...0> with the same random flips (shared by all factors)
            empty = prep.copy_empty_like()
            for rand_index in range(n_randomizations):
                mask = rng.integers(0, 2, self.n).tolist()
                add_circuit(measure_cat(empty, "Z", self.qubits, mask), 0, rand_index, "CAL", mask)

    def shots_per_circuit(self, total_shots: int) -> int:
        return max(1, int(total_shots) // len(self.circuits))

    def _merge(self, counts_list, which):
        """Merge (and un-twirl) counts of all randomizations per (factor, basis)."""
        merged = {}
        for label, counts in zip(self.labels, counts_list):
            if (label["factor"], label["basis"]) not in which:
                continue
            bits, weights = _counts_to_arrays(counts, self.n)
            bits = bits ^ np.asarray(label["mask"], dtype=np.int8)   # undo the random X flips
            key = (label["factor"], label["basis"])
            if key in merged:
                merged[key] = (np.vstack([merged[key][0], bits]), np.concatenate([merged[key][1], weights]))
            else:
                merged[key] = (bits, weights)
        return merged

    def analyze(self, counts_list: Sequence[dict], bootstrap: int = 200, seed: int = 7) -> CatResult:
        """Turn the counts of `self.circuits` into a (mitigated) CatResult.

        Error bars come from a bootstrap: every measured distribution is resampled
        `bootstrap` times, the whole analysis is repeated on each resample, and the
        spread (standard deviation) of the results is reported as the error.
        """
        if len(counts_list) != len(self.circuits):
            raise ValueError("Need exactly one counts dict per circuit, in the same order.")
        recipe = self.recipe
        keys = [(factor, basis) for factor in recipe.zne_factors for basis in ("Z", "X")]
        if recipe.readout_mitigation:
            keys.append((0, "CAL"))
        data = self._merge(counts_list, set(keys))
        rng = np.random.default_rng(seed)

        def estimate(sample):
            """P and X for every noise factor, from one (possibly resampled) data set."""
            P_per_factor, X_per_factor = [], []
            lambdas = None
            if recipe.readout_mitigation:
                cal_bits, cal_weights = sample[(0, "CAL")]
                lambdas = ((1.0 - 2.0 * cal_bits) * cal_weights[:, None]).sum(axis=0) / cal_weights.sum()
                if sample is data and np.any(lambdas < 0.3):
                    bad = [self.qubits[i] for i in np.where(lambdas < 0.3)[0]]
                    warnings.warn(
                        f"Readout of qubit(s) {bad} is almost random (fidelity factor < 0.3). "
                        "Readout mitigation will explode the error bars — avoid these qubits!",
                        stacklevel=3,
                    )
            for factor in recipe.zne_factors:
                bits_z, weights_z = sample[(factor, "Z")]
                bits_x, weights_x = sample[(factor, "X")]
                if recipe.readout_mitigation:
                    P, X, _, _ = _mitigated_estimates(bits_z, weights_z, bits_x, weights_x, lambdas)
                else:
                    all_equal = bits_z.min(axis=1) == bits_z.max(axis=1)
                    P = float((all_equal * weights_z).sum() / weights_z.sum())
                    X = float(((1.0 - 2.0 * (bits_x.sum(axis=1) % 2)) * weights_x).sum() / weights_x.sum())
                P_per_factor.append(P)
                X_per_factor.append(X)
            return P_per_factor, X_per_factor

        P_per_factor, X_per_factor = estimate(data)
        # bootstrap: resample every measured distribution
        boot_P, boot_X = [], []
        for _ in range(bootstrap):
            sample = {}
            for key, (bits, weights) in data.items():
                resampled = rng.multinomial(int(weights.sum()), weights / weights.sum()).astype(float)
                sample[key] = (bits, resampled)
            P_values, X_values = estimate(sample)
            boot_P.append(P_values)
            boot_X.append(X_values)
        boot_P, boot_X = np.array(boot_P), np.array(boot_X)
        P_err_per_factor = boot_P.std(axis=0)
        X_err_per_factor = boot_X.std(axis=0)

        factors, kind = recipe.zne_factors, recipe.zne_extrapolator
        if len(factors) > 1:
            P, P_err = _fit_extrapolate(factors, P_per_factor, P_err_per_factor, kind)
            X, X_err = _fit_extrapolate(factors, X_per_factor, X_err_per_factor, kind)
            # propagate the errors by repeating the extrapolation on every bootstrap sample
            extrapolated = np.array(
                [
                    (
                        _fit_extrapolate(factors, P_values, P_err_per_factor, kind)[0],
                        _fit_extrapolate(factors, X_values, X_err_per_factor, kind)[0],
                    )
                    for P_values, X_values in zip(boot_P, boot_X)
                ]
            )
            F_boot = extrapolated[:, 0] + (extrapolated[:, 1] - 1) / 2
            P_err, X_err = float(extrapolated[:, 0].std()), float(extrapolated[:, 1].std())
            F_err = float(F_boot.std())
        else:
            P, X = P_per_factor[0], X_per_factor[0]
            P_err, X_err = float(P_err_per_factor[0]), float(X_err_per_factor[0])
            F_err = float((boot_P[:, 0] + (boot_X[:, 0] - 1) / 2).std())
        F = P + (X - 1) / 2
        shots = int(sum(sum(c.values()) for c in counts_list))
        details = {
            "per_factor": {
                int(factor): {"P": P_per_factor[i], "X": X_per_factor[i], "F": P_per_factor[i] + (X_per_factor[i] - 1) / 2}
                for i, factor in enumerate(factors)
            },
            "recipe": asdict(recipe),
        }
        return CatResult(F, F_err, P, P_err, X, X_err, self.n, shots, details)

    def run(self, runner, total_shots: int = 8000, bootstrap: int = 200) -> CatResult:
        """Convenience: run all circuits with `runner` (CatSimulator or HardwareRunner) and analyze."""
        counts = runner.run(self.circuits, shots=self.shots_per_circuit(total_shots))
        self.last_counts = counts
        return self.analyze(counts, bootstrap=bootstrap)


def build_cat_experiment(prep, backend, recipe: Recipe | None = None, qubits=None, seed: int = 0) -> CatExperiment:
    """Create the CatExperiment for a cat-preparation circuit (no measurements)."""
    return CatExperiment(prep, backend, recipe or Recipe(), qubits=qubits, seed=seed)


def measure_cat_quality(prep, runner, recipe: Recipe | None = None, total_shots: int = 8000,
                        qubits=None, seed: int = 0) -> CatResult:
    """One-liner: build the experiment for `prep`, run it on `runner`, return the CatResult."""
    exp = build_cat_experiment(prep, runner.backend, recipe, qubits=qubits, seed=seed)
    return exp.run(runner, total_shots=total_shots)


# =============================================================================
# 7. Hardware
# =============================================================================
class HardwareRunner:
    """Run circuits on a real IBM backend (or a fake one) through the Sampler primitive.

    Uses the new client-side Sampler (qiskit_ibm_runtime.executor_sampler, the
    recommended API since qiskit-ibm-runtime 0.50). If that is unavailable or fails,
    set legacy=True to use the (deprecated, still working) SamplerV2.

    Twirling and DD are switched OFF on the server side because the recipe
    already put them into the circuits explicitly — so the very same circuits run
    on the simulator and on the QPU.
    """

    def __init__(self, backend, legacy: bool = False):
        self.backend = backend
        self.legacy = legacy
        self.last_job = None

    def _sampler(self):
        if self.legacy:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
        else:
            from qiskit_ibm_runtime.executor_sampler import Sampler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sampler = Sampler(mode=self.backend)
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        sampler.options.dynamical_decoupling.enable = False
        return sampler

    def submit(self, circuits: Sequence[QuantumCircuit], shots: int):
        """Submit ONE job containing all circuits. Returns the job (save job.job_id()!)."""
        sampler = self._sampler()
        job = sampler.run(list(circuits), shots=int(shots))
        self.last_job = job
        return job

    @staticmethod
    def counts_from_result(result) -> list[dict]:
        return [pub.data.meas.get_counts() for pub in result]

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[dict]:
        """Submit and wait. On a real QPU this can sit in the queue for a while!"""
        job = self.submit(circuits, shots)
        return self.counts_from_result(job.result())


def estimate_qpu_seconds(circuits: QuantumCircuit | Sequence[QuantumCircuit], shots: int, backend,
                         overhead_s: float = 2.0) -> float:
    """Rough estimate of the QPU time (in seconds) a job will be billed for.

    Follows IBM's guide "Estimate job run time":
        usage ≈ overhead per job + (rep_delay + circuit duration) × (circuits × shots)
    with rep_delay = the backend's default (250 µs on Heron) and the circuit
    duration taken from the backend's gate times. Treat it as a ballpark figure:
    look at `job_usage_seconds(job)` after the job to see what was really billed.
    """
    circuits = [circuits] if isinstance(circuits, QuantumCircuit) else list(circuits)
    rep_delay = getattr(backend, "default_rep_delay", None) or 250e-6
    total = 0.0
    for circuit in circuits:
        try:
            duration = schedule_circuit(circuit, backend).estimate_duration(backend.target, unit="s")
        except Exception:
            duration = 100e-6  # IBM's "typical circuit" figure
        total += (rep_delay + duration) * int(shots)
    return overhead_s + total


def job_usage_seconds(job) -> float | None:
    """QPU seconds a finished job was billed for (None if the service doesn't report it)."""
    try:
        usage = job.usage()
        return float(usage) if usage is not None else None
    except Exception:
        try:
            return float(job.metrics()["usage"]["quantum_seconds"])
        except Exception:
            return None


def retrieve_job_counts(job_id: str, **service_kwargs) -> list[dict]:
    """Fetch the counts of a finished hardware job later, e.g. after a kernel restart."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    job = QiskitRuntimeService(**service_kwargs).job(job_id)
    return HardwareRunner.counts_from_result(job.result())


# =============================================================================
# 8. Verification
# =============================================================================
def verify_cat(prep: QuantumCircuit, qubits: Sequence[int] | None = None) -> bool:
    """Check (noiselessly, with a Clifford simulation) that `prep` really makes a cat.

    The ideal output must be stabilized by X...X and by Z_i Z_(i+1) on the cat
    qubits. Works for hundreds of qubits.
    """
    qubits = list(qubits) if qubits is not None else cat_qubits(prep)
    act = _active_qubits([prep])
    act = sorted(set(act) | set(qubits))
    mp = {p: i for i, p in enumerate(act)}
    small = QuantumCircuit(len(act))
    for inst in prep.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        if inst.operation.name in ("measure", "reset") or inst.clbits:
            raise ValueError("verify_cat expects a preparation circuit without measurements.")
        small.append(inst.operation, [mp[prep.find_bit(q).index] for q in inst.qubits])
    try:
        state = StabilizerState(Clifford(small))
    except Exception as exc:  # non-Clifford gate
        raise ValueError(f"Cannot verify: circuit is not Clifford ({exc}).") from None
    n = len(act)

    def pauli(label_by_qubit):
        s = ["I"] * n
        for q, p in label_by_qubit.items():
            s[mp[q]] = p
        return Pauli("".join(reversed(s)))

    if abs(state.expectation_value(pauli({q: "X" for q in qubits})) - 1) > 1e-9:
        return False
    for a, b in zip(qubits[:-1], qubits[1:]):
        if abs(state.expectation_value(pauli({a: "Z", b: "Z"})) - 1) > 1e-9:
            return False
    return True


# =============================================================================
# 9. Submission
# =============================================================================
def _circuit_to_b64(circ: QuantumCircuit) -> str:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return base64.b64encode(buf.getvalue()).decode()


def circuit_from_b64(s: str) -> QuantumCircuit:
    return qpy.load(io.BytesIO(base64.b64decode(s)))[0]


def save_circuit(circuit: QuantumCircuit, path: str) -> str:
    """Save a circuit (including its layout) to a .qpy file — do this for your best cats!"""
    with open(path, "wb") as f:
        qpy.dump(circuit, f)
    return path


def load_circuit(path: str) -> QuantumCircuit:
    with open(path, "rb") as f:
        return qpy.load(f)[0]


def experiment_record(exp: CatExperiment, counts: Sequence[dict], result: CatResult) -> dict:
    """Everything needed to re-analyse an experiment later (by you or by the organizers)."""
    return {
        "prep_qpy_b64": _circuit_to_b64(exp.prep),
        "qubits": exp.qubits,
        "recipe": asdict(exp.recipe),
        "seed": exp.seed,
        "counts": [dict(c) for c in counts],
        "result": _result_dict(result),
    }


def experiment_from_record(rec: dict, backend) -> CatExperiment:
    """Rebuild a CatExperiment (same circuits, same random masks) from experiment_record()."""
    recipe = Recipe(**rec["recipe"])
    prep = circuit_from_b64(rec["prep_qpy_b64"])
    return CatExperiment(prep, backend, recipe, qubits=rec["qubits"], seed=rec["seed"])


# -----------------------------------------------------------------------------
# Scoring helpers (same formulas the organizers use)
# -----------------------------------------------------------------------------
def conservative_error(result: CatResult) -> float:
    """eps* = |1 - F| + sigma_F : distance to a perfect cat, penalised by its uncertainty."""
    return abs(1.0 - result.F) + result.F_err


def error_reduction(baseline: CatResult, team: CatResult) -> float:
    """R = |1 - F_baseline| / eps*_team  (how many times closer to a perfect cat you got)."""
    return abs(1.0 - baseline.F) / max(conservative_error(team), 1e-6)


def reduction_points(R: float, max_points: float, R_full: float = 16.0) -> float:
    """Points for an error-reduction factor R: 0 at R<=1, full marks at R >= R_full (log scale)."""
    if R <= 1:
        return 0.0
    return float(max_points * min(1.0, math.log(R) / math.log(R_full)))


def scorecard(prep: QuantumCircuit, recipe: Recipe, simulator: CatSimulator | None = None,
              total_shots: int = ESTIMATE_SHOTS, qubits=None, seed: int = 11) -> dict:
    """Estimate your Cleanest-Cat simulator score the way the organizers compute it.

    The organizers use a secret simulator seed and SCORING_SHOTS (100,000) shots,
    so their error bars are about 2x smaller than with the default 24,000 here."""
    sim = simulator or CatSimulator()
    if len(qubits or cat_qubits(prep)) != CLEAN_CAT_SIZE:
        raise ValueError(f"The Cleanest Cat has exactly {CLEAN_CAT_SIZE} qubits.")
    if not verify_cat(prep, qubits):
        raise ValueError("This circuit does not prepare a GHZ state (verify_cat failed).")
    base = measure_cat_quality(baseline_cat(sim.backend), sim, Recipe(), total_shots=total_shots, seed=seed)
    team = measure_cat_quality(prep, sim, recipe, total_shots=total_shots, qubits=qubits, seed=seed + 1)
    R = error_reduction(base, team)
    return {"baseline": base, "team": team, "eps_star": conservative_error(team), "R": R,
            "points (of 25)": round(reduction_points(R, 25), 1)}


# -----------------------------------------------------------------------------
# Final hardware job: organizers' baseline + your cat, in ONE job
# -----------------------------------------------------------------------------
class FinalJob:
    """Your final hardware run. The baseline cat runs in the SAME job as your cat, so
    both see the same calibration and the same noise — that makes the comparison fair
    no matter when your job runs.

        final = FinalJob(my_prep, my_recipe, backend, total_shots=40_000)
        job = HardwareRunner(backend).submit(final.circuits, final.shots)
        print(job.job_id())            # write it down!
        ...
        counts = HardwareRunner.counts_from_result(job.result())   # or retrieve_job_counts(job_id)
        base, team = final.analyze(counts)
        record = final.record(job.job_id(), counts)
    """

    def __init__(self, team_prep, team_recipe: Recipe, backend, total_shots: int = 40_000,
                 n: int = CLEAN_CAT_SIZE, seed: int = 2026, team_qubits=None):
        self.backend = backend
        self.base_exp = build_cat_experiment(baseline_cat(backend, n), backend, Recipe(), seed=seed)
        self.team_exp = build_cat_experiment(team_prep, backend, team_recipe, qubits=team_qubits, seed=seed + 1)
        if self.team_exp.n != n:
            raise ValueError(f"Your cat has {self.team_exp.n} qubits, the baseline {n}.")
        n_team = len(self.team_exp.circuits)
        self.shots = max(100, int(0.75 * total_shots) // n_team)
        # give the baseline ~25% of the shots by repeating its two circuits
        self.base_repeats = max(1, round(0.25 * total_shots / (2 * self.shots)))
        self.circuits = self.base_exp.circuits * self.base_repeats + self.team_exp.circuits
        self.n_base = 2 * self.base_repeats

    @property
    def total_shots(self) -> int:
        return self.shots * len(self.circuits)

    def _split(self, counts: Sequence[dict]):
        base_counts = []
        for k in range(2):  # merge the repeats of Z and of X
            merged: dict = {}
            for rep in range(self.base_repeats):
                for key, v in counts[rep * 2 + k].items():
                    merged[key] = merged.get(key, 0) + v
            base_counts.append(merged)
        return base_counts, list(counts[self.n_base:])

    def analyze(self, counts: Sequence[dict]) -> tuple[CatResult, CatResult]:
        base_counts, team_counts = self._split(counts)
        return self.base_exp.analyze(base_counts), self.team_exp.analyze(team_counts)

    def save(self, path: str, job_id: str | None = None) -> str:
        """Save everything needed to analyse the job later (survives kernel restarts)."""
        data = {
            "backend": self.backend.name,
            "job_id": job_id,
            "shots": self.shots,
            "base_repeats": self.base_repeats,
            "base": {"prep_qpy_b64": _circuit_to_b64(self.base_exp.prep), "qubits": self.base_exp.qubits,
                     "recipe": asdict(self.base_exp.recipe), "seed": self.base_exp.seed},
            "team": {"prep_qpy_b64": _circuit_to_b64(self.team_exp.prep), "qubits": self.team_exp.qubits,
                     "recipe": asdict(self.team_exp.recipe), "seed": self.team_exp.seed},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    @classmethod
    def load(cls, path: str, backend) -> "FinalJob":
        """Rebuild a FinalJob saved with .save() (pass the same backend object type/name)."""
        with open(path) as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.backend = backend
        obj.base_exp = experiment_from_record(data["base"], backend)
        obj.team_exp = experiment_from_record(data["team"], backend)
        obj.shots = data["shots"]
        obj.base_repeats = data["base_repeats"]
        obj.circuits = obj.base_exp.circuits * obj.base_repeats + obj.team_exp.circuits
        obj.n_base = 2 * obj.base_repeats
        obj.job_id = data.get("job_id")
        return obj

    def record(self, job_id: str, counts: Sequence[dict]) -> dict:
        """The dict to pass as `hardware=` to save_submission()."""
        base_counts, team_counts = self._split(counts)
        base, team = self.analyze(counts)
        return {
            "backend": self.backend.name,
            "job_id": job_id,
            "baseline": experiment_record(self.base_exp, base_counts, base),
            "team": experiment_record(self.team_exp, team_counts, team),
            "error_reduction": error_reduction(base, team),
        }


def biggest_cat_record(exp: CatExperiment, counts: Sequence[dict], result: CatResult, where: str,
                       job_id: str | None = None) -> dict:
    """The dict to pass as `biggest_cat=` to save_submission(). where = 'hardware' or 'simulator'.

    Biggest Cat rule: the certificate must come from raw data or readout mitigation
    (optionally with DD / gate twirling), NOT from zero-noise extrapolation. ZNE
    estimates what a noiseless chip would give, while the Biggest Cat asks whether
    the state on the real chip is entangled.
    """
    if len(tuple(exp.recipe.zne_factors)) > 1:
        raise ValueError("Biggest Cat: ZNE is not allowed for certification. Re-run the big cat with a recipe "
                         "without zne_factors (raw or readout_mitigation=True, optionally dd / gate_twirling).")
    rec = experiment_record(exp, counts, result)
    rec.update({"n": result.n, "where": where, "backend": exp.backend.name, "job_id": job_id,
                "certified": result.certified})
    return rec


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
def qubit_quality(backend) -> tuple[dict, dict]:
    """Calibration data in plain dicts: ({qubit: {'readout','t1','t2'}}, {(a, b): two-qubit error})."""
    t = backend.target
    two_q = [n for n in ("cz", "ecr", "cx") if n in t.operation_names][0]
    qubits = {}
    for q in range(backend.num_qubits):
        p = t["measure"].get((q,))
        props = t.qubit_properties[q] if t.qubit_properties else None
        qubits[q] = {
            "readout": (p.error if p is not None and p.error is not None else float("nan")),
            "t1": getattr(props, "t1", None),
            "t2": getattr(props, "t2", None),
        }
    edges = {}
    for qs, p in t[two_q].items():
        if p is not None and p.error is not None:
            key = tuple(sorted(qs))
            edges[key] = max(edges.get(key, 0.0), p.error)
    return qubits, edges


def show_chip(backend, prep: QuantumCircuit | None = None, qubits: Sequence[int] | None = None,
              ax=None, figsize=(9, 8)):
    """Draw the processor: qubits coloured by readout error (green = good, red = bad),
    broken couplers in red, your cat's qubits labelled by their index in the cat and
    the couplers your circuit uses in black."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from qiskit.visualization import plot_gate_map

    qinfo, einfo = qubit_quality(backend)
    if qubits is None and prep is not None:
        qubits = cat_qubits(prep)
    qubits = list(qubits or [])
    used_edges = set()
    if prep is not None:
        for inst in _two_qubit_ops(prep):
            used_edges.add(tuple(sorted(prep.find_bit(q).index for q in inst.qubits)))
    cmap = colormaps["RdYlGn_r"]
    colors = []
    for q in range(backend.num_qubits):
        e = qinfo[q]["readout"]
        x = 0.0 if math.isnan(e) else min(1.0, math.log10(max(e, 1e-3) / 1e-3) / 2)  # 0.1% .. 10%
        rgba = np.array(cmap(x))
        if qubits and q not in qubits:  # fade everything that is not part of the cat
            rgba[:3] = 0.35 * rgba[:3] + 0.65
        colors.append(matplotlib_hex(rgba))
    if qubits:
        labels = [str(qubits.index(q)) if q in qubits else "" for q in range(backend.num_qubits)]
    else:
        labels = [str(q) for q in range(backend.num_qubits)]
    line_colors = []
    for a, b in backend.coupling_map.get_edges():
        key = tuple(sorted((a, b)))
        if key in used_edges:
            line_colors.append("#000000")
        elif einfo.get(key, 0) >= 0.5:
            line_colors.append("#ff0000")
        else:
            line_colors.append("#cccccc")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plot_gate_map(backend, qubit_color=colors, qubit_labels=labels, line_color=line_colors,
                  font_color="black", ax=ax)
    ax.set_title("qubit colour = readout error (green ≈ 0.1%, yellow ≈ 1%, red ≥ 10%)\n"
                 "black lines = couplers your cat uses, red lines = broken couplers; "
                 "labels = position in the cat" if qubits else
                 "qubit colour = readout error (green ≈ 0.1%, yellow ≈ 1%, red ≥ 10%); "
                 "red lines = broken couplers; labels = physical qubit index", fontsize=9)
    return ax.figure


def matplotlib_hex(rgba) -> str:
    r, g, b = (int(round(255 * c)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def save_submission(
    path: str,
    team: str,
    members: Sequence[str],
    clean_cat_prep: QuantumCircuit,
    clean_cat_recipe: Recipe,
    clean_cat_sim_result: CatResult | None = None,
    hardware: dict | None = None,
    biggest_cat: dict | None = None,
    answers: dict | None = None,
    techniques: Sequence[str] = (),
    clean_cat_qubits: Sequence[int] | None = None,
) -> str:
    """Write your team's submission file (JSON). Upload it together with your notebook.

    clean_cat_prep   : your N=12 cat-preparation circuit for fake_kingston (ISA, no measurements)
    clean_cat_recipe : the Recipe the organizers should use to measure it
    hardware         : FinalJob.record(...) of your final hardware job (optional, worth points)
    biggest_cat      : biggest_cat_record(...) (optional)
    answers          : your short answers {"Q1": "...", ...}
    techniques       : list of techniques you used and can back up with data
    """
    qubits = list(clean_cat_qubits) if clean_cat_qubits is not None else cat_qubits(clean_cat_prep)
    if len(qubits) != CLEAN_CAT_SIZE:
        raise ValueError(f"The Cleanest Cat must have exactly {CLEAN_CAT_SIZE} qubits (yours: {len(qubits)}).")
    if not verify_cat(clean_cat_prep, qubits):
        raise ValueError("Your clean-cat circuit does not prepare a GHZ state. Check it with verify_cat().")
    payload = {
        "format": "herding-cats/1",
        "toolkit_version": __version__,
        "team": team,
        "members": list(members),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clean_cat": {
            "prep_qpy_b64": _circuit_to_b64(clean_cat_prep),
            "qubits": qubits,
            "recipe": asdict(clean_cat_recipe),
            "self_reported_sim": None if clean_cat_sim_result is None else _result_dict(clean_cat_sim_result),
        },
        "hardware": hardware,
        "biggest_cat": biggest_cat,
        "techniques": list(techniques),
        "answers": answers or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return path


def _result_dict(r: CatResult) -> dict:
    return {"F": r.F, "F_err": r.F_err, "P": r.P, "P_err": r.P_err, "X": r.X, "X_err": r.X_err,
            "n": r.n, "shots": r.shots}


