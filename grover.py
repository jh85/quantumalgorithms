import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def main():
    target_state = "10" * 10
    n = len(target_state)

    Uw = QuantumCircuit(n)
    lendian = target_state[::-1]
    for i in range(n):
        if lendian[i] == "0":
            Uw.x(i)
    Uw.h(n-1)
    Uw.mcx(list(range(n-1)), n-1)
    Uw.h(n-1)
    for i in range(n):
        if lendian[i] == "0":
            Uw.x(i)

    Us = QuantumCircuit(n)
    Us.h(range(n))
    Us.x(range(n))
    Us.h(n-1)
    Us.mcx(list(range(n-1)), n-1)
    Us.h(n-1)
    Us.x(range(n))
    Us.h(range(n))

    grover_step = QuantumCircuit(n, name="Grover Step")
    grover_step.compose(Uw, inplace=True)
    grover_step.compose(Us, inplace=True)
    optimal_itr = math.floor(math.pi / 4 * math.sqrt(2**n))

    qc = QuantumCircuit(n,n)
    qc.h(range(n))
    for _ in range(optimal_itr):
        qc.compose(grover_step, inplace=True)
    qc.measure(range(n), range(n))
    simulator = AerSimulator(device="GPU")
    compiled_circuit = transpile(qc, simulator)

    print(f"Grover iteration: {optimal_itr}")
    print("--- Circuit Statistics ---")
    print(f"Logical Qubits: {qc.num_qubits}")
    print(f"Logical Depth: {qc.depth()}")
    print(f"Transpiled Qubits: {compiled_circuit.num_qubits}")
    print(f"Transpiled Depth: {compiled_circuit.depth()}")

    shots = 1000
    print(f"Running the explicit simulation ({shots} shots)...")
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts(compiled_circuit)
    print(counts)

main()
