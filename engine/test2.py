import numpy as np


confusion = np.zeros((num_classes, num_classes))
rows = targets.cpu().numpy()