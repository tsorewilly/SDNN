import torch
import torch.nn as nn

class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()
        #print(input_size)

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, 1),   #1
            nn.ReLU(),    #激励层
            nn.Dropout(),

            #nn.Conv1d(64, 64, 5),
            #nn.ReLU(),
            #nn.Dropout(),

            nn.Conv1d(64, 64, 2),   #2
            nn.ReLU(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(4608, 128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128, num_classes),
            )

    def forward(self, x):
        #print(x.size())  #1*1*56
        x = self.features(x)
        #print(x.size())  #1*64*44
        x = x.view(x.size(0), 4608)
        #print(x.size())  #1*2816
        out = self.classifier(x)
        #print(out.size()) #1*3
        return out