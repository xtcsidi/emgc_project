import subprocess
import sys
import os
import time
import threading

def predownload_mnist():
    """Ensure MNIST dataset is downloaded before booting subprocesses to avoid race conditions."""
    print("Pre-downloading MNIST dataset if not present...")
    import torch
    from torchvision import datasets, transforms
    data_path = "./data"
    os.makedirs(data_path, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    datasets.MNIST(data_path, train=True, download=True, transform=transform)
    datasets.MNIST(data_path, train=False, download=True, transform=transform)
    print("MNIST dataset is ready!")

def stream_logs(prefix, process, color_code):
    """Asynchronously stream stdout/stderr of a node with styled terminal colors."""
    reset_code = "\033[0m"
    try:
        for line in iter(process.stdout.readline, b""):
            decoded_line = line.decode('utf-8', errors='replace').strip()
            if decoded_line:
                print(f"{color_code}{prefix}{reset_code} | {decoded_line}")
    except Exception:
        pass

import argparse

def run_cluster():
    parser = argparse.ArgumentParser(description="Decentralized P2P Cluster Launcher")
    parser.add_argument("--num_nodes", type=int, default=3, help="Number of nodes to spawn (2 to 10)")
    args = parser.parse_args()
    
    num_nodes = args.num_nodes
    if num_nodes < 2 or num_nodes > 10:
        print("Please choose a number of nodes between 2 and 10.")
        return
        
    # 1. Download MNIST
    predownload_mnist()
    
    # Enable colored logs in terminal
    os.system("") 
    
    COLORS = [
        "\033[96m",  # Cyan
        "\033[95m",  # Purple
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[94m",  # Blue
        "\033[91m",  # Red
        "\033[36m",  # Dark Cyan
        "\033[35m",  # Dark Purple
        "\033[32m",  # Dark Green
        "\033[33m",  # Dark Yellow
    ]
    
    node_configs = []
    base_port = 9001
    
    for i in range(num_nodes):
        node_id = f"node-{i}"
        port = base_port + i
        
        # Connect to all other active peers in the cluster
        peer_ports = [str(base_port + j) for j in range(num_nodes) if j != i]
        peers_str = ",".join(peer_ports)
        
        color = COLORS[i % len(COLORS)]
        
        node_configs.append({
            "node_id": node_id,
            "port": port,
            "peers": peers_str,
            "shard": i,
            "color": color
        })
    
    print("\n=======================================================")
    print(f"  LAUNCHING DECENTRALIZED P2P WORKSTATION CLUSTER ({num_nodes} NODES)  ")
    print("=======================================================\n")
    
    processes = []
    threads = []
    
    for config in node_configs:
        cmd = [
            sys.executable,
            "-u",
            "node.py",
            "--node_id", config["node_id"],
            "--port", str(config["port"]),
            "--peers", config["peers"],
            "--shard_index", str(config["shard"]),
            "--rounds", "4",
            "--total_shards", str(num_nodes)
        ]
        
        # Start node subprocess
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
        processes.append(p)
        
        # Start thread to capture log output
        t = threading.Thread(
            target=stream_logs,
            args=(f"[{config['node_id']}]", p, config["color"]),
            daemon=True
        )
        t.start()
        threads.append(t)
        
        # Stagger boots slightly to prevent port clashes
        time.sleep(1.0)
        
    print("\nNodes booted! Running P2P Gossip Training on local shards...")
    
    try:
        # Wait for all nodes to complete cleanly
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n[Master] Interrupted. Terminating node processes...")
        for p in processes:
            p.terminate()
            
    print("\n=======================================================")
    print("         P2P CLUSTER TRAIN COMPLETED CLEANLY           ")
    print("=======================================================\n")

if __name__ == "__main__":
    run_cluster()
