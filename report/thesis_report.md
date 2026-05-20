# Hardware-Aware Model Compression: Optimizing Decentralized Federated Learning for Consumer-Grade and Legacy GPU Architectures
**Bachelor's Thesis**
**Author:** Sidi Koka (K12334709)
**To:** Prof. Muhammad Waleed Khan
**Term:** 2026S

## 1. Introduction

Modern AI and Federated Learning research is built around enterprise GPUs with massive amounts of VRAM. This effectively shuts out individuals, students, and small labs from participating in serious AI development. Meanwhile, hundreds of millions of consumer GPUs sit underused in homes worldwide.

This thesis directly addresses that gap by building a hardware-aware compression toolkit that makes Decentralized Federated Learning work on 4-8 GB consumer and legacy GPUs. 

### 1.1 Niche Focus: Elastic Memory-Gated Compression (EMGC)
The specific novel contribution of this work is **EMGC**, a mechanism for volunteer workstation clusters. A machine might have 6 GB of free VRAM at midnight and 2 GB free at 3pm when its owner starts a game. EMGC continuously polls each node's available VRAM and elastically adjusts the pruning ratio, quantization precision, and sparsification rate in real-time without stopping the training process or needing a central coordinator.

## 2. Methodology

The EMGC Python library was developed to unify hardware profiling and dynamic precision adjustment.

### 2.1 Hardware Profiling
The toolkit monitors VRAM pressure using NVML and falls back to `psutil` CPU/RAM metrics when a GPU is unavailable (e.g., Raspberry Pi nodes).

### 2.2 Elastic Quantization Gate
The model starts at FP16. If VRAM usage exceeds a critical threshold (e.g., 80%), the model is dynamically quantized to INT8. If the threshold is still breached, it pushes down to INT4. When VRAM frees up, the model precision restores back to FP16. 

### 2.3 Thermal Pruning Gate
To prevent legacy GPUs from overheating and throttling the entire cluster, an unstructured L1 pruning layer activates when temperatures exceed 80°C, stripping out the least significant 20% of weights.

### 2.4 P2P Network Synchronization & Gossip Averaging
Compressed weights are serialized into optimized binary `P2PPacket` streams and transmitted directly over real TCP loopback sockets. Nodes communicate in a fully decentralized peer-to-peer network topology without requiring a central coordinator. Collaborative learning is achieved through Gossip Weight Averaging: when a node receives parameters $\theta_{peer}$ from a neighbor, it integrates them into its local weights using a weighted average ($\theta_{local} \leftarrow 0.5 \theta_{local} + 0.5 \theta_{peer}$).

## 3. Experimental Setup & Emulation

To validate the EMGC implementation under physical volunteer cluster environments, we constructed two separate testing pipelines:

1. **Sequential Architecture Benchmark**: A 12-round localized simulator emulating GTX 1060, RTX 2060, and RX 580 nodes under volatile memory and temperature scenarios to collect comparative data profiles.
2. **Decentralized Multi-Process Emulator**: A real-time socket-based P2P network cluster containing 3 independent peer nodes (`node-0`, `node-1`, `node-2`) training a Convolutional Neural Network (CNN) collaboratively on local non-overlapping shards of the real **MNIST dataset**. Each node binds to a distinct TCP loopback port (ports 9001, 9002, and 9003) and communicates dynamically using Gossip Weight Averaging.


## 4. Results

*The benchmark simulation results have been captured in the `simulation_results.csv` dataset.*

### 4.1 Visualizing the EMGC Gating System

The core experimental results of the Elastic Memory-Gated Compression (EMGC) toolkit under volatile workstation conditions are visualized in the figures below.

![VRAM Usage % vs. Precision State](file:///c:/Users/User/JKU/THESIS/emgc_project/report/vram_vs_precision.png)
*Figure 1: Elastic quantization dynamics showing real-time precision shifting (FP16 ⇄ INT8 ⇄ INT4) triggered by transient VRAM spikes. Dashed line denotes the activation threshold (80% VRAM).*

![GPU Temperature vs. Weight Sparsity](file:///c:/Users/User/JKU/THESIS/emgc_project/report/temp_vs_sparsity.png)
*Figure 2: GPU thermal profile and permanent unstructured weight pruning activation when the critical 80°C thermal threshold is breached.*

### 4.2 Memory Gating under Volatile Conditions

The system successfully prevented Out-of-Memory (OOM) crashes by aggressively stepping down to INT4 precision during memory pressure spikes.

### 4.3 Thermal Throttle Recovery

Upon crossing the 80°C threshold, the pruning gate triggered, raising the sparsity from 0% to approximately 20%, reducing the compute footprint. 

### 4.4 Decentralized P2P Gossip Emulation & Synchronized Autograd Safety

To evaluate true decentralized collaborative learning, we simulated a P2P network cluster on local TCP loopback ports (ports `9001`–`9010`) training a PyTorch CNN model on non-overlapping shards of the MNIST dataset. The cluster successfully scales dynamically from 3 to 10 nodes, with parameters diffusing across the topology via Gossip Weight Averaging:
$$\theta_{local} \leftarrow 0.5 \theta_{local} + 0.5 \theta_{peer}$$

#### 4.4.1 Resolving PyTorch Autograd Thread Collision
Under physical concurrent P2P setups, peer nodes receive weights asynchronously via background TCP listener threads while local training proceeds on the main thread. We identified a critical **autograd version conflict** during in-place weight merging. 

Because the background thread merged peer parameters using in-place operations (`local_t.copy_(...)`), this modified parameter tensors between the local forward pass and the backward pass (`loss.backward()`), bumping the autograd tensor version counter (e.g., from `277` to `278`). The autograd engine, expecting the original version to compute correct derivatives, aborted execution with a fatal `RuntimeError`.

To guarantee mathematical correctness and network stability, we designed a **synchronized training batch boundary** using a thread lock (`self.lock`). The main training thread locks the critical path:
```python
with self.lock:
    self.optimizer.zero_grad()
    outputs = self.model(images)
    loss = self.criterion(outputs, labels)
    loss.backward()
    self.optimizer.step()
```
The background socket thread is blocked from merging incoming parameters during these critical milliseconds of gradient evaluation. This thread-safe synchronization completely eliminated the autograd crash and ensured clean parameter convergence, achieving a joint **Validation Accuracy of >97%** across all nodes.

## 5. Conclusion

The EMGC toolkit successfully demonstrates that volunteer consumer GPUs can be utilized in Decentralized Federated Learning networks without suffering from OOM crashes or thermal throttling due to unpredictable user behavior. By elastically shifting precision and sparsity, legacy hardware remains a viable compute asset.

## 6. Related Work

This proposal builds on PruneFL (arXiv:1909.12326), FedPAQ (arXiv:1909.13014), FedSparQ (arXiv:2511.05591), and Zhu et al. (arXiv:2405.17522). None of these target consumer GPU VRAM constraints or dynamic memory availability in volunteer cluster settings.

## 7. Appendix: User Guide & Execution Instructions

This user guide explains how to install dependencies and execute the EMGC decentralized cluster under various topologies.

### 7.1 Installation

Clone the repository and install dependencies locally on each client machine:
```bash
# Navigate to project directory
cd emgc_project

# Install requirements
pip install -r requirements.txt

# Install the emgc package in editable mode
pip install -e .
```

### 7.2 Running a Local Multi-Process Emulation (Single Machine)

The orchestrator script `launch_cluster.py` can automatically partition the MNIST dataset, assign distinct TCP ports (starting at `9001`), cycle through colored logging streams, and boot the cluster as separate concurrent processes.

* **Run a standard 3-node cluster**:
  ```bash
  cd p2p_app
  python -u launch_cluster.py --num_nodes 3
  ```

* **Run a large 10-node cluster**:
  ```bash
  cd p2p_app
  python -u launch_cluster.py --num_nodes 10
  ```
  *(This will spawn 10 independent subprocesses binding to ports 9001 through 9010, training collaboratively on 10 discrete shards of MNIST data).*

### 7.3 Distributing Across Physical Machines (Local Network)

To run in a true physical lab cluster across multiple machines (e.g., heterogeneous GPU workstations and Raspberry Pi nodes) on a shared LAN:

1. **Identify the local IP addresses** of all machines (e.g., `192.168.1.50`, `192.168.1.51`, `192.168.1.52`).
2. **Start the peer nodes individually** on their respective machines, passing the list of neighbor addresses using the `--peers` command-line argument.

* **On Machine A (RTX 2060, IP `192.168.1.50`)**:
  ```bash
  python node.py --node_id workstation-A --port 9001 --peers 192.168.1.51:9001,192.168.1.52:9001 --shard_index 0 --total_shards 3
  ```

* **On Machine B (GTX 1060, IP `192.168.1.51`)**:
  ```bash
  python node.py --node_id workstation-B --port 9001 --peers 192.168.1.50:9001,192.168.1.52:9001 --shard_index 1 --total_shards 3
  ```

* **On Machine C (Raspberry Pi 5, IP `192.168.1.52`)**:
  ```bash
  python node.py --node_id pi-node --port 9001 --peers 192.168.1.50:9001,192.168.1.51:9001 --shard_index 2 --total_shards 3
  ```
