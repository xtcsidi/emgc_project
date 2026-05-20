import sys
import os
import socket
import threading
import time
import json
import struct
import io
import argparse
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.progress import track
    RICH_AVAILABLE = True
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
except ImportError:
    RICH_AVAILABLE = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

log = logging.getLogger("P2PNode")

# Add parent directory to path to import emgc package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emgc.optimizer import MemoryGatedOptimizer
from emgc.datatypes import Precision, HardwareSnapshot, P2PPacket
import emgc.quantization as quant
quant._BNB_AVAILABLE = False  # Use robust PyTorch fake-quant fallback for CPU/consumer compatibility

# ── 1. Lightweight PyTorch CNN for MNIST ──────────────────────────────────────
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── 2. Socket Helper Functions ────────────────────────────────────────────────
def recv_exact(sock, num_bytes):
    """Receive exactly num_bytes from a TCP socket."""
    data = b""
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            return None
        data += packet
    return data

# ── 3. Simulated Hardware Profiles (to demonstrate volatility) ───────────────
# We simulate VRAM spikes and high temperatures in different rounds
NODE_SCENARIOS = {
    "node-0": [
        {"vram": 1000, "temp": 45},  # Rd 1: FP16
        {"vram": 5500, "temp": 50},  # Rd 2: VRAM spike -> INT8/INT4
        {"vram": 5800, "temp": 85},  # Rd 3: VRAM high + Temp high -> Pruning 20%
        {"vram": 1500, "temp": 55},  # Rd 4: Recovers
    ],
    "node-1": [
        {"vram": 1200, "temp": 40},  # Rd 1: FP16
        {"vram": 1500, "temp": 88},  # Rd 2: Temp high -> Pruning 20%
        {"vram": 1800, "temp": 60},  # Rd 3: Steady
        {"vram": 1500, "temp": 50},  # Rd 4: Steady
    ],
    "node-2": [
        {"vram": 1500, "temp": 45},  # Rd 1: FP16
        {"vram": 2000, "temp": 50},  # Rd 2: FP16 (RX 580 style, high VRAM ceiling)
        {"vram": 2200, "temp": 55},  # Rd 3: FP16
        {"vram": 6500, "temp": 92},  # Rd 4: VRAM spike + Temp high -> compression + pruning
    ]
}

# ── 4. Main P2P Node Implementation ───────────────────────────────────────────
class P2PNode:
    def __init__(self, node_id, port, peers, dataset_shard_index, total_shards=3):
        self.node_id = node_id
        self.port = port
        self.peers = peers  # List of integers (ports)
        self.dataset_shard_index = dataset_shard_index
        self.total_shards = total_shards
        
        # Load local model
        self.model = MNISTCNN()
        
        # Simulated hardware snapshot values
        self.sim_vram_total = 6144.0  # 6GB VRAM
        self.sim_vram_used = 1200.0   # Baseline VRAM
        self.sim_temp = 45.0          # Baseline Temperature
        
        # Custom subclassed optimizer to mock hardware polling
        class SimulatedNodeOptimizer(MemoryGatedOptimizer):
            def __init__(self, outer_self, *args, **kwargs):
                self.outer = outer_self
                super().__init__(*args, **kwargs)
                
            def profile_hardware(self):
                snap = HardwareSnapshot()
                snap.gpu_available = True
                snap.vram_total_mb = self.outer.sim_vram_total
                snap.vram_used_mb = self.outer.sim_vram_used
                snap.vram_pct = self.outer.sim_vram_used / self.outer.sim_vram_total
                snap.gpu_temp_c = self.outer.sim_temp
                return snap
                
        self.emgc_optimizer = SimulatedNodeOptimizer(
            outer_self=self,
            model=self.model,
            node_id=self.node_id,
            vram_threshold=0.80,
            temp_threshold=80.0,
            prune_amount=0.20
        )
        
        # Capture optimizer device
        self.device = self.emgc_optimizer.device
        
        # Model standard configurations
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        
        # Load partitions of real MNIST dataset
        self.train_loader, self.test_loader = self._prepare_data()
        
        # Lock for thread-safety during weight updates
        self.lock = threading.Lock()
        
        # Socket server shutdown event
        self.stop_event = threading.Event()

    def _prepare_data(self):
        """Load MNIST and extract a specific subset shard for this peer node."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Ensure data directory exists
        data_path = "./data"
        os.makedirs(data_path, exist_ok=True)
        
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        
        # Split train dataset into shards
        total_len = len(train_dataset)
        shard_size = total_len // self.total_shards
        indices = list(range(total_len))
        
        # Non-overlapping partition index
        start_idx = self.dataset_shard_index * shard_size
        end_idx = start_idx + shard_size
        shard_indices = indices[start_idx:end_idx]
        
        local_train_subset = Subset(train_dataset, shard_indices)
        
        train_loader = DataLoader(local_train_subset, batch_size=64, shuffle=True)
        # Use full validation set to test aggregate performance
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        return train_loader, test_loader

    # ── Background TCP Server for Receiving Weights ───────────────────────────
    def start_socket_server(self):
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        log.info(f"[{self.node_id}] Listening for P2P connections on port {self.port}...")

    def _run_server(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", self.port))
        server_sock.listen(5)
        server_sock.settimeout(1.0)
        
        while not self.stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue
            
            client_thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            client_thread.start()
            
        server_sock.close()

    def _handle_client(self, conn):
        """Read and parse custom binary P2PPacket layout."""
        try:
            # 1. Read 32-byte header
            header_bytes = recv_exact(conn, 32)
            if not header_bytes:
                return
            
            magic = header_bytes[0:4]
            if magic != b"MGOP":
                log.error(f"[{self.node_id}] Received invalid packet magic.")
                return
                
            version = struct.unpack("<I", header_bytes[4:8])[0]
            bits = struct.unpack("<I", header_bytes[8:12])[0]
            arch_id = struct.unpack("<I", header_bytes[12:16])[0]
            sender_id = header_bytes[16:32].decode('ascii').strip("\x00")
            
            # 2. Read 4-byte metadata length
            meta_len_bytes = recv_exact(conn, 4)
            if not meta_len_bytes:
                return
            meta_len = struct.unpack("<I", meta_len_bytes)[0]
            
            # 3. Read metadata json
            meta_bytes = recv_exact(conn, meta_len)
            metadata = json.loads(meta_bytes.decode())
            
            # 4. Read payload (remaining model weights bytes)
            # The remaining stream is the payload bytes. 
            # We read until connection closes or we read exact payload size from network.
            # PyTorch weights can vary. We'll read all available remaining stream.
            payload_data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                payload_data += chunk
                
            # 5. Extract weights & Gossip Average
            buf = io.BytesIO(payload_data)
            remote_state = torch.load(buf, map_location="cpu", weights_only=True)
            
            with self.lock:
                local_state = self.model.state_dict()
                merged_count = 0
                for key in local_state:
                    if key in remote_state:
                        # Real Gossip Weight Averaging: merge local and remote parameters (50% each)
                        # We convert remote weight tensor back to float/half depending on local type
                        local_t = local_state[key]
                        remote_t = remote_state[key].to(local_t.device)
                        
                        # Handle size mismatches if any
                        if local_t.shape == remote_t.shape:
                            local_t.copy_(0.5 * local_t + 0.5 * remote_t)
                            merged_count += 1
                
                log.info(
                    f"◀◀ [{self.node_id}] Integrated weights from '{sender_id}' via Gossip Avg! "
                    f"({merged_count} tensors merged, Precision: {metadata['precision']}, Peer Sparsity: {metadata['sparsity']:.1%})"
                )
                
        except Exception as e:
            log.error(f"[{self.node_id}] Exception in handling remote weights: {e}")
        finally:
            conn.close()

    # ── P2P Gossip Broadcasting ───────────────────────────────────────────────
    def gossip_to_peer(self, peer):
        """Serialize state and transmit over TCP socket to peer."""
        if isinstance(peer, tuple):
            peer_ip, peer_port = peer
        else:
            peer_ip = "127.0.0.1"
            peer_port = int(peer)
            
        try:
            # 1. Generate P2PPacket via EMGC controller
            with self.lock:
                packet = self.emgc_optimizer.export_weights_p2p()
                
            serialized_bytes = packet.serialize()
            
            # 2. Open TCP connection to target IP and port
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_sock.connect((peer_ip, peer_port))
            
            # 3. Write bytes and close
            client_sock.sendall(serialized_bytes)
            client_sock.close()
            
            log.info(
                f"▶▶ [{self.node_id}] Transmitted P2PPacket ({len(serialized_bytes) / 1024:.1f} KB) "
                f"to peer {peer_ip}:{peer_port}. Precision: {packet.precision.value}, Sparsity: {packet.metadata['sparsity']:.1%}"
            )
        except Exception as e:
            log.warning(f"[{self.node_id}] Gossip link to {peer_ip}:{peer_port} failed: {e}")

    # ── Local Training Loop ───────────────────────────────────────────────────
    def train_epoch(self, epoch):
        self.model.train()
        correct = 0
        total = 0
        loss_val = 0.0
        
        # Profile hardware and trigger elastic constraints before training phase
        snap = self.emgc_optimizer.profile_hardware()
        with self.lock:
            self.emgc_optimizer.apply_elastic_quantization(snap)
            self.emgc_optimizer.apply_thermal_pruning(snap)
            
        loader_iter = self.train_loader
        if RICH_AVAILABLE:
            loader_iter = track(self.train_loader, description=f"[bold cyan]Training Epoch {epoch}...[/bold cyan]")
            
        for images, labels in loader_iter:
            images, labels = images.to(self.device), labels.to(self.device)
            with self.lock:
                self.optimizer.zero_grad()
                
                # If compressed down to FP16/INT8/INT4 fallback, cast inputs to half precision
                if self.emgc_optimizer.current_precision != Precision.FP32:
                    images = images.half()
                    # Ensure model parameter types match input types (done automatically by self.model.half())
                    
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            loss_val += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = correct / total
        log.info(
            f"[{self.node_id}] Epoch {epoch} complete | "
            f"Local Train Loss: {loss_val/len(self.train_loader):.4f} | "
            f"Local Acc: {acc:.2%}"
        )
        return acc

    def validate(self):
        """Evaluate local model on general validation set."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if self.emgc_optimizer.current_precision != Precision.FP32:
                    images = images.half()
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        log.info(f"[{self.node_id}] [EVAL] Validation Accuracy (MNIST Test Set): {val_acc:.2%}")
        return val_acc

    def stop(self):
        self.stop_event.set()

# ── 5. Main CLI Entrypoint ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2P Training Peer Node")
    parser.add_argument("--node_id", type=str, required=True, help="Unique string node ID")
    parser.add_argument("--port", type=int, required=True, help="TCP port to bind node server")
    parser.add_argument("--peers", type=str, default="", help="Comma-separated list of peer ports")
    parser.add_argument("--shard_index", type=int, required=True, help="Index of MNIST data subset shard")
    parser.add_argument("--rounds", type=int, default=4, help="Number of simulated training rounds")
    parser.add_argument("--total_shards", type=int, default=3, help="Total number of nodes/shards in the network")
    args = parser.parse_args()
    
    peer_list = []
    if args.peers:
        for p in args.peers.split(","):
            p = p.strip()
            if not p:
                continue
            if ":" in p:
                parts = p.split(":")
                peer_ip = parts[0].strip()
                peer_port = int(parts[1].strip())
                peer_list.append((peer_ip, peer_port))
            else:
                peer_list.append(("127.0.0.1", int(p)))
    
    node = P2PNode(
        node_id=args.node_id,
        port=args.port,
        peers=peer_list,
        dataset_shard_index=args.shard_index,
        total_shards=args.total_shards
    )
    
    # Start background TCP receiver
    node.start_socket_server()
    time.sleep(2)  # Allow port binding to stabilize
    
    try:
        # Load volatility scenario sequence
        scenario = NODE_SCENARIOS.get(args.node_id, [{"vram": 1000, "temp": 45}] * args.rounds)
        
        for rnd in range(1, args.rounds + 1):
            if RICH_AVAILABLE:
                console.rule(f"[bold magenta]--- [{args.node_id}] ROUND {rnd}/{args.rounds} ---[/bold magenta]")
            else:
                print(f"\n--- [{args.node_id}] ROUND {rnd}/{args.rounds} ---")
            
            # Inject dynamic hardware fluctuations for this round
            state = scenario[rnd - 1] if rnd <= len(scenario) else scenario[-1]
            node.sim_vram_used = state["vram"]
            node.sim_temp = state["temp"]
            
            if RICH_AVAILABLE:
                table = Table(title=f"Node: {args.node_id} | Status Dashboard", box=box.ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="bold yellow")
                table.add_row("VRAM Used", f"{state['vram']} MB")
                table.add_row("Temperature", f"{state['temp']} °C")
                table.add_row("Active Peers", str(len(node.peers)))
                for p in node.peers:
                    table.add_row(" ↳ Peer Address", f"{p[0]}:{p[1]}")
                console.print(table)
            else:
                log.info(f"[{args.node_id}] Current HW State: VRAM={state['vram']}MB, Temp={state['temp']}C")
            
            # 1. Train local epoch on real data partition
            node.train_epoch(rnd)
            
            # 2. Validate current state
            node.validate()
            
            # 3. Gossip: exchange compressed weights with randomized/available peer
            if node.peers:
                import random
                target_peer = random.choice(node.peers)
                node.gossip_to_peer(target_peer)
                
            time.sleep(4)  # Small wait to let threads digest gossip packets
            
    except KeyboardInterrupt:
        log.info("Interrupted, shutting down.")
    finally:
        node.stop()
        print(f"\n[{args.node_id}] Node execution completed cleanly.")
