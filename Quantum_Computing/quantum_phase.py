# -*- coding: utf-8 -*-

"""
    Code written entirely by Eric Alcaide: https://github.com/EricAlcaide

    Quantum Phase Estimation, one of the most important subroutines in quantum computation.
    It serves as a central building block for many quantum algorithms
    and implements a measurement for essentially any Hermitian operator.
    Check the math @ IBM's Quantum Experience Guide

    I am using IBM's quantum computing SDK and API in python.
    You can get it here: https://github.com/IBM/qiskit-sdk-py

"""

import sys
sys.path.append("../../qiskit-sdk-py")
from qiskit import QuantumProgram
from tools import visualization
import Qconfig

# Create the QuantumProgram object instance.
n = 2
QPS_SPECS = {
    "circuits": [{
        "name": "qc",
        "quantum_registers": [{
            "name": "qr", "size": n
        }],
        "classical_registers": [
            {"name": "cr", "size": n}
        ]}]
}
qp = QuantumProgram(specs=QPS_SPECS)

# Get the circuit by Name
circuit = qp.get_circuit("qc")
# Get the registers by Name
qRegister = circuit.regs['qr']
cRegister = circuit.regs['cr']

# Apply the H-gate to QuBITS 0,1 .
circuit.h(qRegister[0])
circuit.h(qRegister[1])

# Apply Z to QuBIT 0 and C-NOT to 1,0
circuit.z(qRegister[0])
circuit.cx(qRegister[1], qRegister[0])

# Now, we apply the H-gate to QuBIT 1
circuit.h(qRegister[1])

# That's it for this algorithm! Measure the qubits into the classical registers.
circuit.measure(qRegister[1], cRegister[1])

# Set the API
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])
# Backend to execute your program, in this case it is the online simulator
device = 'ibmqx_qasm_simulator'
# Group of circuits to execute
circuits = ["qc"]  
# Execute the program
result = qp.execute(circuits, backend=device,
                    coupling_map=None, shots=1024)
print(result)
print(result.get_counts(circuits[0]))
visualization.plot_histogram(result.get_counts(circuits[0]))