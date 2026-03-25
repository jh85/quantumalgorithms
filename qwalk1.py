from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

def main():
    # --- 1. The 9-Qubit Architecture ---
    move_X = QuantumRegister(3, name='move_x')
    move_O = QuantumRegister(3, name='move_o')
    is_X_6 = QuantumRegister(1, name='is_x_6')
    is_O_6 = QuantumRegister(1, name='is_o_6')
    phase = QuantumRegister(1, name='phase')
    
    # We only measure the 6 move qubits
    c_moves = ClassicalRegister(6, name='c_moves')

    qc = QuantumCircuit(move_X, move_O, is_X_6, is_O_6, phase, c_moves)

    # Initialize phase target to |-> for kickback
    qc.x(phase)
    qc.h(phase)

    # The Quantum Coin Toss
    qc.h(move_X)
    qc.h(move_O)

    # ==========================================
    # --- GROVER LOOP (Run exactly 1 time) -----
    # ==========================================
    
    # A. Targeted Oracle (Compute)
    qc.x(move_X[0])
    qc.mcx(list(move_X), is_X_6)
    qc.x(move_X[0])
    
    qc.x(move_O[0])
    qc.mcx(list(move_O), is_O_6)
    qc.x(move_O[0])
    
    # B. Phase Kickback
    qc.x(is_O_6)
    qc.mcx([is_X_6, move_O[2], is_O_6], phase)
    qc.x(is_O_6)
    
    # C. Targeted Oracle (Uncompute)
    qc.x(move_O[0])
    qc.mcx(list(move_O), is_O_6)
    qc.x(move_O[0])
    
    qc.x(move_X[0])
    qc.mcx(list(move_X), is_X_6)
    qc.x(move_X[0])

    # D. Grover Diffuser
    moves = list(move_X) + list(move_O)
    qc.h(moves)
    qc.x(moves)
    qc.h(moves[-1])
    qc.mcx(moves[:-1], moves[-1])
    qc.h(moves[-1])
    qc.x(moves)
    qc.h(moves)

    # ==========================================
    # --- END GROVER LOOP ----------------------
    # ==========================================

    # --- 3. Measurement ---
    qc.measure(move_X, c_moves[0:3])
    qc.measure(move_O, c_moves[3:6])

    # --- 4. Execution on Real Hardware ---
    print("Authenticating with IBM Quantum...")

    # service = QiskitRuntimeService()
    # backend = service.least_busy(simulator=False, operational=True)

    # Specify the instance to fix the warning
    service = QiskitRuntimeService(instance="XXXX")
    # backend = service.backend("ibm_fez")
    backend = service.backend("ibm_marrakesh")

    
    print(f"Target Backend: {backend.name}")
    
    # Transpile the circuit for this specific hardware's topology
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_qc = pm.run(qc)
    
    print(f"Logical Depth: {qc.depth()}")
    print(f"Transpiled Depth: {isa_qc.depth()} (Fingers crossed!)")

    print("\nSending to queue...")
    sampler = SamplerV2(mode=backend)
    
    # NEW V2 SYNTAX: Set shots in the options before running
    sampler.options.default_shots = 1000 
    job = sampler.run([isa_qc])
    
    print(f"Job ID: {job.job_id()}")
    print("Waiting for results (this may take a few minutes in the queue)...")
    
    result = job.result()[0]
    counts = result.data.c_moves.get_counts()
    
    # Sort results by probability
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    print("\n--- Top 10 Hardware Results ---")
    for bitstring, count in sorted_counts[:10]:
        move_O_bits = bitstring[0:3]
        move_X_bits = bitstring[3:6]
        
        sq_X = int(move_X_bits, 2)
        sq_O = int(move_O_bits, 2)
        print(f"X played {sq_X}, O played {sq_O} | Occurrences: {count}/1000")


main()
