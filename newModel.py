import torch
import torch.nn as nn

class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, 5),   #1
            nn.BatchNorm1d(64),
            nn.ReLU(),    #激励层
            #nn.AvgPool1d(2, stride=2), #You can only use this or MaxPool2d
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(),

            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv1d(64, 64, 5),   #2
            nn.ReLU(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        	nn.Dropout(),
        	nn.Linear(1728, 128),        #根据数据的大小，需要改64
        	nn.ReLU(),                   #According to the size of the data, need to change 64
        	nn.Dropout(),
        	nn.Linear(128, num_classes),
        	)

    def forward(self, x):
    	x = self.features(x);
    	x = x.view(x.size(0), 1728) #根据数据的大小，需要改64,只是特征数对其有影响，行数多少没影响
    	out = self.classifier(x)    #According to the size of the data, it needs to be changed to 64,
                                    # but the number of features affects it, the number of rows has no effect

    	return out