import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


"""
Deutsch-Jozsa Algorithm - 30 Qubit Implementation
===================================================
Circuit layout:
  - N input qubits  (q[0] … q[N-1])
  - 1  ancilla qubit (q[N])

The algorithm decides if a black-box function f: {0,1}^N -> {0,1} is
  - CONSTANT: f(x) = 0 for all x, or f(x) = 1 for all x
  - BALANCED: f(x) = 0 for exactly half of all inputs, 1 for the other half

Classically this requires up to 2^(N-1) + 1 queries.
The Deutsch-Jozsa algorithm does it in a single query.

Measurement outcome:
  - All-zeros on the input register -> CONSTANT
  - Any non-zero bit                -> BALANCED
"""


N_INPUT = 10          # number of input qubits
SHOTS   = 1000        # one shot is theoretically sufficient; use more to verify

def constant_oracle_zero(qc: QuantumCircuit, input_qubits, ancilla) -> None:
    """
    Constant oracle: f(x) = 0 for all x.
    Implementation: Do nothing — the ancilla is never flipped, so U_f|x⟩|y⟩ = |x⟩|y ⊕ 0⟩ = |x⟩|y⟩
    """
    pass

def constant_oracle_one(qc: QuantumCircuit, input_qubits, ancilla) -> None:
    """
    Constant oracle: f(x) = 1 for all x.
    Implementation: Flip the ancilla unconditionally with an X gate: U_f|x⟩|y⟩ = |x⟩|y ⊕ 1⟩
    """
    qc.x(ancilla)

def balanced_oracle(qc: QuantumCircuit, input_qubits, ancilla) -> None:
    """
    Balanced oracle: f(x) = parity(x) = x_0 ⊕ x_1 ⊕ … ⊕ x_{n-1}.
    Exactly half of all 2^N inputs have even parity (f=0) and half have odd parity (f=1), so the function is balanced.
    Implementation: Apply a CNOT from every input qubit to the ancilla. Each CNOT flips the ancilla iff the control qubit is |1⟩,
    realising the XOR chain.  This is the canonical balanced Deutsch-Jozsa oracle.
    U_f|x⟩|y⟩ = |x⟩|y ⊕ (x_0 ⊕ x_1 ⊕ … ⊕ x_{n-1})⟩
    """
    for q in input_qubits:
        qc.cx(q, ancilla)

def balanced_oracle_rand(qc: QuantumCircuit, input_qubits, ancilla, seed: int = 0) -> None:
    """
    Alternative balanced oracle using a random subset of input qubits.
    A random subset S ⊆ {0,…,n-1} is chosen (with |S| ≥ 1).
    f(x) = ⊕_{i∈S} x_i  is still balanced as long as S is non-empty.
    Additionally, a random bit-string b is used to flip selected inputs before the CNOT (phase kick-back trick),
    giving more variety without breaking balance.
    """
    # Choose a non-empty subset of input qubits to XOR
    n = len(input_qubits)
    mask = np.random.choice([0,1], size=n)  # 0 or 1 for each qubit
    if mask.sum() == 0:                # guarantee at least one CNOT
        mask[0] = 1

    # Random input bit-flips (pre- and post-) to disguise the oracle
    flip_bits = np.random.choice([0,1], size=n)
    for i, q in enumerate(input_qubits):
        if flip_bits[i]:
            qc.x(q)

    for i, q in enumerate(input_qubits):
        if mask[i]:
            qc.cx(q, ancilla)

    # Undo the input flips (oracle must not change the input register)
    for i, q in enumerate(input_qubits):
        if flip_bits[i]:
            qc.x(q)


def build_deutsch_jozsa(n: int, oracle_fn) -> QuantumCircuit:
    """
    Build the full (n+1)-qubit Deutsch-Jozsa circuit.
    Steps
    -----
    1. Initialise ancilla to |1⟩ via X gate.
    2. Apply H to all n input qubits and to the ancilla
       -> input register in uniform superposition |+⟩^n
       -> ancilla in |−⟩ = (|0⟩−|1⟩)/√2  (phase-kickback target)
    3. Apply the oracle U_f.
    4. Apply H again to all n input qubits.
    5. Measure the input register.
    Outcome
    -------
    |0…0⟩ -> constant
    anything else -> balanced
    """
    input_reg   = QuantumRegister(n,  name='input')
    ancilla_reg = QuantumRegister(1,  name='ancilla')
    meas_reg    = ClassicalRegister(n, name='meas')

    qc = QuantumCircuit(input_reg, ancilla_reg, meas_reg)

    # ── Step 1: put ancilla in |1⟩ ──────────────────────────────────────────
    qc.x(ancilla_reg[0])
    qc.barrier()

    # ── Step 2: Hadamard on everything ──────────────────────────────────────
    qc.h(input_reg)
    qc.h(ancilla_reg[0])
    qc.barrier()

    # ── Step 3: Oracle ───────────────────────────────────────────────────────
    oracle_fn(qc, input_reg, ancilla_reg[0])
    qc.barrier()

    # ── Step 4: Hadamard on input register ───────────────────────────────────
    qc.h(input_reg)
    qc.barrier()

    # ── Step 5: Measure input register ───────────────────────────────────────
    qc.measure(input_reg, meas_reg)

    return qc

def main():
    # ── Oracle 1: Constant f(x) = 0
    qc0 = build_deutsch_jozsa(N_INPUT, constant_oracle_zero)
    # ── Oracle 2: Constant f(x) = 1
    qc1 = build_deutsch_jozsa(N_INPUT, constant_oracle_one)
    # ── Oracle 3: Balanced  f(x) = parity(x)
    qc2 = build_deutsch_jozsa(N_INPUT, balanced_oracle)
    # ── Oracle 4: Balanced  f(x) = random subset XOR
    qc3 = build_deutsch_jozsa(N_INPUT, balanced_oracle_rand)

    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    for qc in [qc1,qc1,qc2,qc3]:
        tqc = transpile(qc, backend=backend, optimization_level=0) # optimization_level=0 is necessary for balanced oracles

        print("--- Circuit Statistics ---")
        print(f"Logical Qubits: {qc.num_qubits}")
        print(f"Logical Depth: {qc.depth()}")
        print(f"Transpiled Qubits: {tqc.num_qubits}")
        print(f"Transpiled Depth: {tqc.depth()}")
        
        sampler = SamplerV2(mode=backend)
        shots = 1000
        job =sampler.run([tqc], shots=shots)
        result = job.result()[0]
        idx = 0
        creg_name = tqc.cregs[idx].name
        counts = result.data[creg_name].get_counts()
        print(counts)

    for qc in [qc0,qc1,qc2,qc3]:
        simulator  = AerSimulator(method='statevector')
        #simulator  = AerSimulator(method='stabilizer')
        transpiled = transpile(qc, simulator, optimization_level=0)
        job     = simulator.run(transpiled, shots=SHOTS)
        result  = job.result()
        counts = result.get_counts()
        print(counts)
        print(f"  Qubits : {qc.num_qubits}")
        print(f"  Depth  : {qc.depth()}")
        print(f"  Gates  : {qc.count_ops()}")

main()
