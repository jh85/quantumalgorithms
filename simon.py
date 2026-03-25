import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

def create_simon_oracle(s):
    n = len(s)
    qc = QuantumCircuit(2 * n + 1, name="Simon_Oracle")
    
    # Qiskit is little endian
    # Qiskit: |q9 q8 q7 ... q1 q0>
    # s:      "q0 q1 ... q7 q8 q9"
    s_rev = s[::-1]

    # 1. Copy the input register to the output register
    for i in range(n):
        qc.cx(i, i + n)

    # 2. Apply the 2-to-1 mapping for the hidden string
    first_one = s_rev.find("1")
    for i in range(n):
        if s_rev[i] == "1":
            qc.cx(first_one, n + i)
    return qc

def solve_for_s_noisy(counts: dict, n: int) -> str:
    # 1. Sort measurements by frequency (highest count to lowest count)
    sorted_strings = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)    
    equations = []

    # 2. Iterate through the sorted strings and build the matrix dynamically
    for bitstring in sorted_strings:
        # Ignore the all-zero string
        if bitstring == "0" * n:
            continue

        # Convert to integer list (reversed for Qiskit's endianness)
        new_row = [int(bit) for bit in bitstring[::-1]]
        equations.append(new_row)
        
        # 3. Perform GF(2) row reduction on our current set of equations
        M = np.array(equations, dtype=int)
        rows, cols = M.shape
        
        r = 0
        pivots = []
        for c in range(cols):
            pivot_row = -1
            for i in range(r, rows):
                if M[i, c] == 1:
                    pivot_row = i
                    break
            if pivot_row == -1:
                continue
            
            # Swap rows to bring the pivot to the current row 'r'
            M[[r, pivot_row]] = M[[pivot_row, r]]
            pivots.append(c)
            
            # Eliminate other 1s in the current column
            for i in range(rows):
                if i != r and M[i, c] == 1:
                    M[i] = (M[i] + M[r]) % 2
            r += 1
        
        # 4. Check the rank of the matrix
        # If we have found exactly n - 1 pivots (independent equations), we can stop
        # The noise lower down the list can no longer interfere.
        if len(pivots) == n - 1:
            print(f"Success! Found enough independent equations after checking {len(equations)} top bitstrings.")
            
            # Identify the free variable
            free_vars = [c for c in range(cols) if c not in pivots]
            if len(free_vars) != 1:
                return "0" * n # Trivial case fallback
                
            free_var = free_vars[0]
            s_array = np.zeros(cols, dtype=int)
            s_array[free_var] = 1 # Set free variable to 1
            
            # Back-substitute to find the rest of s
            for i, c in enumerate(pivots):
                s_array[c] = M[i, free_var]
                
            # Convert back to a string and reverse to match original endianness
            return ''.join(map(str, s_array))[::-1]

    # If the loop finishes and we never hit n - 1 pivots, the noise was too high.
    return "Error: Could not find enough independent equations in the data."

def main():
    # 1. Define the n-bit secret string
    secret_s = "100101"
    n = len(secret_s)
    shots = int(2**(n-1) * 30)
    print(f"n={n} shots={shots}")

    qr_in = QuantumRegister(n, "in")
    qr_out = QuantumRegister(n, "out")
    sw = QuantumRegister(1, "sw")
    cr = ClassicalRegister(n, "meas")
    qc = QuantumCircuit(qr_in, qr_out, sw, cr)

    # Apply Hadamard to input register
    qc.h(qr_in)
    
    # Apply the oracle
    oracle = create_simon_oracle(secret_s)
    qc.compose(oracle, inplace=True)
    
    # Apply Hadamard to input register again
    qc.h(qr_in)
    
    # Measure the input register
    qc.measure(qr_in, cr)

    # Execute the circuit
    print(f"Executing Simon's Algorithm for hidden string: {secret_s}")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    tqc = transpile(qc, backend=backend)
    sampler = SamplerV2(mode=backend)
    job =sampler.run([tqc], shots=shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    
    print(sorted(counts.items(), key=lambda itm:itm[1], reverse=True)[:n])
    
    estimated_s = solve_for_s_noisy(counts, n)
    print(f"Estimated hidden string:       {estimated_s}")
    print(f"Algorithm successful?          {secret_s == estimated_s}")

main()
