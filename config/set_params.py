import torch

class params:
    """Modify this class to set parameters."""
    def __init__(self):
        self.params = {
            "root": "data/data_pig_train1.csv",
            "test": "data/data_pig_test1.csv",
            "resume": None,
            "input_dim": 1,
            "num_classes": 3,
            "workers": 4,
            "batch_size": 16,
            "epochs": 200,
            "lr": 0.003,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "split": 0.2,
            "use_cuda": torch.cuda.is_available(),
            "Checkpoint": "Checkpoint1",
            "a": 1 ,
            "val_acc": "0.2",
            "test_acc": "0.2"
        }