import torchvision.models as models
import torch
#from torch import nn
from ptflops import get_model_complexity_info
#import torch.nn.functional as F

with torch.cuda.device(0):
  net = models.alexnet()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))



