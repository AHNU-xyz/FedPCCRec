import numpy as np
import random
import torch
def init_seed(seed):
   
    np.random.seed(seed)

   
    random.seed(seed)

    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  