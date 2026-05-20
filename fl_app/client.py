import sys
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import flwr as fl
from collections import OrderedDict
import logging

sys.path.append("../")
from emgc.optimizer import MemoryGatedOptimizer

class EMGCFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, optimizer):
        self.model = model
        self.emgc_optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # 1. Profile and trigger EMGC memory/thermal gating
        snap = self.emgc_optimizer.profile_hardware()
        print(f"[{self.emgc_optimizer.node_id}] Hardware snapshot before training: {snap}")
        
        self.emgc_optimizer.apply_elastic_quantization(snap)
        self.emgc_optimizer.apply_thermal_pruning(snap)
        
        # 2. Mock training step
        self.model.train()
        for _ in range(2):
            inputs = torch.randn(2, 3, 224, 224) 
            # In fp16, int8, or int4, the weights are half precision for computation
            if self.emgc_optimizer.current_precision.value in ["fp16", "int8", "int4"]:
                inputs = inputs.half()
                
            labels = torch.randint(0, 1000, (2,))
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
        print(f"[{self.emgc_optimizer.node_id}] Finished training round.")
        return self.get_parameters(config={}), 2 * 2, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return float(0.5), 100, {"accuracy": float(0.85)}

if __name__ == "__main__":
    node_id = sys.argv[1] if len(sys.argv) > 1 else "client-1"
    
    print(f"Starting {node_id}...")
    model = mobilenet_v2(weights=None)
    
    emgc_opt = MemoryGatedOptimizer(
        model=model,
        node_id=node_id,
        vram_threshold=0.80,
    )
    
    client = EMGCFlowerClient(model, emgc_opt)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
