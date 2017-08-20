# I am using IBM's quantum computing SDK and API in python.
# You can get it here: https://github.com/IBM/qiskit-sdk-py
# To use the API, you need a IBM QX account, which is free at
# https://quantumexperience.ng.bluemix.net/qx

import sys
sys.path.append("../../qiskit-sdk-py") # solve the relative dependencies if you clone QISKit from the Git repo and use like a global.

from qiskit import QuantumProgram

import Qconfig

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = "ibmqx2"
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

###############################################################
# Make a quantum program for the GHZ state.
###############################################################
QPS_SPECS = {
    "circuits": [{
        "name": "ghz",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]}]
}

qp = QuantumProgram(specs=QPS_SPECS)
qc = qp.get_circuit("ghz")
q = qp.get_quantum_register("q")
c = qp.get_classical_register("c")

# Create a GHZ state
qc.h(q[0])
for i in range(4):
    qc.cx(q[i], q[i+1])
# Insert a barrier before measurement
qc.barrier()
# Measure all of the qubits in the standard basis
for i in range(5):
    qc.measure(q[i], c[i])

###############################################################
# Set up the API and execute the program.
###############################################################
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])

# First version: no mapping
print("no mapping, simulator")
result = qp.execute(["ghz"], backend='ibmqx_qasm_simulator',
                    coupling_map=None, shots=1024)
print(result)
print(result.get_counts("ghz"))

# Second version: map to qx2 coupling graph and simulate
print("map to %s, simulator" % backend)
result = qp.execute(["ghz"], backend='ibmqx_qasm_simulator',
                    coupling_map=coupling_map, shots=1024)
print(result)
print(result.get_counts("ghz"))

# Third version: map to qx2 coupling graph and simulate locally
print("map to %s, local qasm simulator" % backend)
result = qp.execute(["ghz"], backend='local_qasm_simulator',
                    coupling_map=coupling_map, shots=1024)
print(result)
print(result.get_counts("ghz"))

# Fourth version: map to qx2 coupling graph and run on qx2
print("map to %s, backend" % backend)
result = qp.execute(["ghz"], backend=backend,
                    coupling_map=coupling_map, shots=1024, timeout=120)
print(result)
print(result.get_counts("ghz"))