import sys
sys.path.append("../")
import csv
import torch.nn as nn
from emgc.optimizer import MemoryGatedOptimizer
from emgc.datatypes import HardwareSnapshot
import emgc.quantization as quant
quant._BNB_AVAILABLE = False

class MockModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

def run_simulation(gpu_name="GTX 1060", vram_total_mb=6144, temp_baseline=40.0):
    model = MockModel()
    
    class SimulatedOptimizer(MemoryGatedOptimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sim_vram_used = 1000.0
            self.sim_temp = temp_baseline
            
        def profile_hardware(self):
            snap = HardwareSnapshot()
            snap.gpu_available = True
            snap.vram_total_mb = vram_total_mb
            snap.vram_used_mb = self.sim_vram_used
            snap.vram_pct = self.sim_vram_used / vram_total_mb
            snap.gpu_temp_c = self.sim_temp
            return snap
            
    optimizer = SimulatedOptimizer(model=model, node_id=f"sim-{gpu_name}")
    
    scenario = [
        {"vram": 1000, "temp": 45},
        {"vram": 1500, "temp": 50},
        {"vram": 2000, "temp": 55},
        {"vram": 5000, "temp": 60}, 
        {"vram": 5200, "temp": 65}, 
        {"vram": 5500, "temp": 70}, 
        {"vram": 6000, "temp": 75}, 
        {"vram": 2000, "temp": 75}, 
        {"vram": 2000, "temp": 85}, 
        {"vram": 2000, "temp": 90},
        {"vram": 2000, "temp": 60},
        {"vram": 2000, "temp": 60},
    ]

    results = []
    
    print(f"\n--- Starting Simulation for {gpu_name} ---")
    for rnd, state in enumerate(scenario, 1):
        optimizer.sim_vram_used = state["vram"]
        optimizer.sim_temp = state["temp"]
        
        snap = optimizer.profile_hardware()
        optimizer.apply_elastic_quantization(snap)
        optimizer.apply_thermal_pruning(snap)
        
        packet = optimizer.export_weights_p2p()
        
        results.append({
            "round": rnd,
            "vram_used_mb": state["vram"],
            "vram_pct": snap.vram_pct,
            "temp_c": state["temp"],
            "precision": packet.precision.value,
            "sparsity": packet.metadata["sparsity"]
        })
        
    return results

if __name__ == "__main__":
    gpus = [
        {"name": "GTX 1060", "vram": 6144},
        {"name": "RTX 2060", "vram": 6144},
        {"name": "RX 580", "vram": 8192},
    ]
    
    all_results = []
    for gpu in gpus:
        res = run_simulation(gpu_name=gpu["name"], vram_total_mb=gpu["vram"])
        for r in res:
            r["gpu"] = gpu["name"]
            all_results.append(r)
            
    csv_file = "simulation_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gpu", "round", "vram_used_mb", "vram_pct", "temp_c", "precision", "sparsity"])
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\nResults saved to {csv_file}")
