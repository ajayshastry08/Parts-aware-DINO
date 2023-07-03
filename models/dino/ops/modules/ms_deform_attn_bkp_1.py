# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import time

from ..functions import MSDeformAttnFunction

def box_cxcywh_to_4_point_split_v3(x,l=0.2,v=0.6):
    #note that x is in the format cxcywh
    mask_new = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]).to("cuda")
    x_new = x[:, :, :,None] * mask_new[None, :]
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    x_xyxy[:,:,:,:,2] = x_new[:,:,:,:,0] + 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,3] = x_new[:,:,:,:,1] + 0.5 * x_new[:,:,:,:,3]
    
    x_xywh[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xywh[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0,0] = x_new[:,:,:,0,0]
    x_final[:,:,:,0,1] = x_new[:,:,:,0,1]
    x_final[:,:,:,0,2] = x_new[:,:,:,0,2]
    x_final[:,:,:,0,3] = x_new[:,:,:,0,3]

    x_final[:,:,:,1,0] = x_new[:,:,:,1,0] 
    x_final[:,:,:,1,1] = x_new[:,:,:,1,1] - x_new[:,:,:,1,3]*l*v
    x_final[:,:,:,1,2] = x_new[:,:,:,1,2]
    x_final[:,:,:,1,3] = x_new[:,:,:,1,3]

    x_final[:,:,:,2,0] = x_new[:,:,:,2,0] - l*(x_new[:,:,:,2,0]-x_xyxy[:,:,:,2,0])
    x_final[:,:,:,2,1] = x_new[:,:,:,2,1] + l*(x_xyxy[:,:,:,2,3]-x_new[:,:,:,2,1])
    x_final[:,:,:,2,2] = x_new[:,:,:,2,2]
    x_final[:,:,:,2,3] = x_new[:,:,:,2,3]

    x_final[:,:,:,3,0] = x_new[:,:,:,3,0] + l*(x_xyxy[:,:,:,3,2]-x_new[:,:,:,3,0])
    x_final[:,:,:,3,1] = x_new[:,:,:,3,1] + l*(x_xyxy[:,:,:,3,3]-x_new[:,:,:,3,1])
    x_final[:,:,:,3,2] = x_new[:,:,:,3,2]
    x_final[:,:,:,3,3] = x_new[:,:,:,3,3]
    
    return x_final #torch.stack(b, dim=-1)
def box_cxcywh_to_5_point_split_v4(x):
    #note that x is in the format cxcywh
    mask_new = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]).to("cuda")
    x_new = x[:, :, :,None] * mask_new[None, :]
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    x_xyxy[:,:,:,:,2] = x_new[:,:,:,:,0] + 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,3] = x_new[:,:,:,:,1] + 0.5 * x_new[:,:,:,:,3]
    
    x_xywh[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xywh[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0,0] = x_xywh[:,:,:,0,0]+x_xywh[:,:,:,0,2]*0.5
    x_final[:,:,:,0,1] = x_xywh[:,:,:,0,1]+x_xywh[:,:,:,0,3]*0.115
    x_final[:,:,:,0,2] = x_xywh[:,:,:,0,2]*0.7
    x_final[:,:,:,0,3] = x_xywh[:,:,:,0,3]*0.23

    x_final[:,:,:,1,0] = x_xywh[:,:,:,1,0] + x_xywh[:,:,:,1,2]*0.25 
    x_final[:,:,:,1,1] = x_xywh[:,:,:,1,1] + x_xywh[:,:,:,1,3]*0.41
    x_final[:,:,:,1,2] = x_xywh[:,:,:,1,2]*0.5
    x_final[:,:,:,1,3] = x_xywh[:,:,:,1,3]*0.36

    x_final[:,:,:,2,0] = x_xywh[:,:,:,2,0] + x_xywh[:,:,:,2,2]*0.75
    x_final[:,:,:,2,1] = x_xywh[:,:,:,2,1] + x_xywh[:,:,:,2,3]*0.41
    x_final[:,:,:,2,2] = x_xywh[:,:,:,2,2]*0.5
    x_final[:,:,:,2,3] = x_xywh[:,:,:,2,3]*0.36

    x_final[:,:,:,3,0] = x_xywh[:,:,:,3,0] + x_xywh[:,:,:,3,2]*0.5
    x_final[:,:,:,3,1] = x_xywh[:,:,:,3,1] + x_xywh[:,:,:,3,3]*0.655
    x_final[:,:,:,3,2] = x_xywh[:,:,:,3,2]*0.7
    x_final[:,:,:,3,3] = x_xywh[:,:,:,3,3]*0.23

    x_final[:,:,:,4,0] = x_xywh[:,:,:,4,0] + x_xywh[:,:,:,4,2]*0.5
    x_final[:,:,:,4,1] = x_xywh[:,:,:,4,1] + x_xywh[:,:,:,4,3]*0.885
    x_final[:,:,:,4,2] = x_xywh[:,:,:,4,2]*0.7
    x_final[:,:,:,4,3] = x_xywh[:,:,:,4,3]*0.23
        
    return x_final 

def box_cxcywh_to_head(x):
    #note that x is in the format cxcywh
    x_new = x
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    x_xyxy[:,:,:,2] = x_new[:,:,:,0] + 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,3] = x_new[:,:,:,1] + 0.5 * x_new[:,:,:,3]
    
    x_xywh[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xywh[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0] = x_xywh[:,:,:,0]+x_xywh[:,:,:,2]*0.5
    x_final[:,:,:,1] = x_xywh[:,:,:,1]+x_xywh[:,:,:,3]*0.115
    x_final[:,:,:,2] = x_xywh[:,:,:,2]*0.7
    x_final[:,:,:,3] = x_xywh[:,:,:,3]*0.23
        
    return x_final 

def box_cxcywh_to_left_torso(x):
    #note that x is in the format cxcywh
    x_new = x
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    x_xyxy[:,:,:,2] = x_new[:,:,:,0] + 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,3] = x_new[:,:,:,1] + 0.5 * x_new[:,:,:,3]
    
    x_xywh[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xywh[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0] = x_xywh[:,:,:,0] + x_xywh[:,:,:,2]*0.25 
    x_final[:,:,:,1] = x_xywh[:,:,:,1] + x_xywh[:,:,:,3]*0.41
    x_final[:,:,:,2] = x_xywh[:,:,:,2]*0.5
    x_final[:,:,:,3] = x_xywh[:,:,:,3]*0.36
        
    return x_final

def box_cxcywh_to_right_torso(x):
    #note that x is in the format cxcywh
    x_new = x
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    x_xyxy[:,:,:,2] = x_new[:,:,:,0] + 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,3] = x_new[:,:,:,1] + 0.5 * x_new[:,:,:,3]
    
    x_xywh[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xywh[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0] = x_xywh[:,:,:,0] + x_xywh[:,:,:,2]*0.75
    x_final[:,:,:,1] = x_xywh[:,:,:,1] + x_xywh[:,:,:,3]*0.41
    x_final[:,:,:,2] = x_xywh[:,:,:,2]*0.5
    x_final[:,:,:,3] = x_xywh[:,:,:,3]*0.36
        
    return x_final  

def box_cxcywh_to_upper_leg(x):
    #note that x is in the format cxcywh
    x_new = x
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    x_xyxy[:,:,:,2] = x_new[:,:,:,0] + 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,3] = x_new[:,:,:,1] + 0.5 * x_new[:,:,:,3]
    
    x_xywh[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xywh[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0] = x_xywh[:,:,:,0] + x_xywh[:,:,:,2]*0.5
    x_final[:,:,:,1] = x_xywh[:,:,:,1] + x_xywh[:,:,:,3]*0.655
    x_final[:,:,:,2] = x_xywh[:,:,:,2]*0.7
    x_final[:,:,:,3] = x_xywh[:,:,:,3]*0.23
        
    return x_final 

def box_cxcywh_to_lower_leg(x):
    #note that x is in the format cxcywh
    x_new = x
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    x_xyxy[:,:,:,2] = x_new[:,:,:,0] + 0.5 * x_new[:,:,:,2]
    x_xyxy[:,:,:,3] = x_new[:,:,:,1] + 0.5 * x_new[:,:,:,3]
    
    x_xywh[:,:,:,0] = x_new[:,:,:,0] - 0.5 * x_new[:,:,:,2]
    x_xywh[:,:,:,1] = x_new[:,:,:,1] - 0.5 * x_new[:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0] = x_xywh[:,:,:,0] + x_xywh[:,:,:,2]*0.5
    x_final[:,:,:,1] = x_xywh[:,:,:,1] + x_xywh[:,:,:,3]*0.885
    x_final[:,:,:,2] = x_xywh[:,:,:,2]*0.7
    x_final[:,:,:,3] = x_xywh[:,:,:,3]*0.23
        
    return x_final 

def box_cxcywh_to_5_point_split_v4(x):
    #note that x is in the format cxcywh
    x_new = x
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    x_xyxy[:,:,:,:,2] = x_new[:,:,:,:,0] + 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,3] = x_new[:,:,:,:,1] + 0.5 * x_new[:,:,:,:,3]
    
    x_xywh[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xywh[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0,0] = x_xywh[:,:,:,0,0]+x_xywh[:,:,:,0,2]*0.5
    x_final[:,:,:,0,1] = x_xywh[:,:,:,0,1]+x_xywh[:,:,:,0,3]*0.115
    x_final[:,:,:,0,2] = x_xywh[:,:,:,0,2]*0.7
    x_final[:,:,:,0,3] = x_xywh[:,:,:,0,3]*0.23

    x_final[:,:,:,1,0] = x_xywh[:,:,:,1,0] + x_xywh[:,:,:,1,2]*0.25 
    x_final[:,:,:,1,1] = x_xywh[:,:,:,1,1] + x_xywh[:,:,:,1,3]*0.41
    x_final[:,:,:,1,2] = x_xywh[:,:,:,1,2]*0.5
    x_final[:,:,:,1,3] = x_xywh[:,:,:,1,3]*0.36

    x_final[:,:,:,2,0] = x_xywh[:,:,:,2,0] + x_xywh[:,:,:,2,2]*0.75
    x_final[:,:,:,2,1] = x_xywh[:,:,:,2,1] + x_xywh[:,:,:,2,3]*0.41
    x_final[:,:,:,2,2] = x_xywh[:,:,:,2,2]*0.5
    x_final[:,:,:,2,3] = x_xywh[:,:,:,2,3]*0.36

    x_final[:,:,:,3,0] = x_xywh[:,:,:,3,0] + x_xywh[:,:,:,3,2]*0.5
    x_final[:,:,:,3,1] = x_xywh[:,:,:,3,1] + x_xywh[:,:,:,3,3]*0.655
    x_final[:,:,:,3,2] = x_xywh[:,:,:,3,2]*0.7
    x_final[:,:,:,3,3] = x_xywh[:,:,:,3,3]*0.23

    x_final[:,:,:,4,0] = x_xywh[:,:,:,4,0] + x_xywh[:,:,:,4,2]*0.5
    x_final[:,:,:,4,1] = x_xywh[:,:,:,4,1] + x_xywh[:,:,:,4,3]*0.885
    x_final[:,:,:,4,2] = x_xywh[:,:,:,4,2]*0.7
    x_final[:,:,:,4,3] = x_xywh[:,:,:,4,3]*0.23
        
    return x_final 

def box_cxcywh_to_4_point_split_8_v3(x,l=0.2,v=0.6):
    #note that x is in the format cxcywh
    #print(x.shape)
    mask_new = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]).to("cuda")
    x_new = x[:, :, :,None] * mask_new[None, :]
    #print(x_new.shape)
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    x_xyxy[:,:,:,:,2] = x_new[:,:,:,:,0] + 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,3] = x_new[:,:,:,:,1] + 0.5 * x_new[:,:,:,:,3]
    
    x_xywh[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xywh[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0,0] = x_new[:,:,:,0,0]
    x_final[:,:,:,0,1] = x_new[:,:,:,0,1]
    x_final[:,:,:,0,2] = x_new[:,:,:,0,2]
    x_final[:,:,:,0,3] = x_new[:,:,:,0,3]

    x_final[:,:,:,1,0] = x_new[:,:,:,1,0] 
    x_final[:,:,:,1,1] = x_new[:,:,:,1,1] - x_new[:,:,:,1,3]*l*v
    x_final[:,:,:,1,2] = x_new[:,:,:,1,2]
    x_final[:,:,:,1,3] = x_new[:,:,:,1,3]

    x_final[:,:,:,2,0] = x_new[:,:,:,2,0] - l*(x_new[:,:,:,2,0]-x_xyxy[:,:,:,2,0])
    x_final[:,:,:,2,1] = x_new[:,:,:,2,1] + l*(x_xyxy[:,:,:,2,3]-x_new[:,:,:,2,1])
    x_final[:,:,:,2,2] = x_new[:,:,:,2,2]
    x_final[:,:,:,2,3] = x_new[:,:,:,2,3]

    x_final[:,:,:,3,0] = x_new[:,:,:,3,0] + l*(x_xyxy[:,:,:,3,2]-x_new[:,:,:,3,0])
    x_final[:,:,:,3,1] = x_new[:,:,:,3,1] + l*(x_xyxy[:,:,:,3,3]-x_new[:,:,:,3,1])
    x_final[:,:,:,3,2] = x_new[:,:,:,3,2]
    x_final[:,:,:,3,3] = x_new[:,:,:,3,3]

    x_final[:,:,:,4,0] = x_new[:,:,:,4,0]
    x_final[:,:,:,4,1] = x_new[:,:,:,4,1]
    x_final[:,:,:,4,2] = x_new[:,:,:,4,2]
    x_final[:,:,:,4,3] = x_new[:,:,:,4,3]

    x_final[:,:,:,5,0] = x_new[:,:,:,5,0] 
    x_final[:,:,:,5,1] = x_new[:,:,:,5,1] - x_new[:,:,:,5,3]*l*v
    x_final[:,:,:,5,2] = x_new[:,:,:,5,2]
    x_final[:,:,:,5,3] = x_new[:,:,:,5,3]

    x_final[:,:,:,6,0] = x_new[:,:,:,6,0] - l*(x_new[:,:,:,6,0]-x_xyxy[:,:,:,6,0])
    x_final[:,:,:,6,1] = x_new[:,:,:,6,1] + l*(x_xyxy[:,:,:,6,3]-x_new[:,:,:,6,1])
    x_final[:,:,:,6,2] = x_new[:,:,:,6,2]
    x_final[:,:,:,6,3] = x_new[:,:,:,6,3]

    x_final[:,:,:,7,0] = x_new[:,:,:,7,0] + l*(x_xyxy[:,:,:,7,2]-x_new[:,:,:,7,0])
    x_final[:,:,:,7,1] = x_new[:,:,:,7,1] + l*(x_xyxy[:,:,:,7,3]-x_new[:,:,:,7,1])
    x_final[:,:,:,7,2] = x_new[:,:,:,7,2]
    x_final[:,:,:,7,3] = x_new[:,:,:,7,3]

    return x_final #torch.stack(b, dim=-1)

def box_cxcywh_to_4_point_split_12_v3(x,l=0.2,v=0.6):
    #note that x is in the format cxcywh
    #print(x.shape)
    mask_new = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]).to("cuda")
    x_new = x[:, :, :,None] * mask_new[None, :]
    #print(x_new.shape)
    x_xyxy = x_new.clone().detach()
    x_xywh = x_new.clone().detach()
    x_final = x_new.clone().detach()

    x_xyxy[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    x_xyxy[:,:,:,:,2] = x_new[:,:,:,:,0] + 0.5 * x_new[:,:,:,:,2]
    x_xyxy[:,:,:,:,3] = x_new[:,:,:,:,1] + 0.5 * x_new[:,:,:,:,3]
    
    x_xywh[:,:,:,:,0] = x_new[:,:,:,:,0] - 0.5 * x_new[:,:,:,:,2]
    x_xywh[:,:,:,:,1] = x_new[:,:,:,:,1] - 0.5 * x_new[:,:,:,:,3]
    
    #modifying first row of the tensor
    x_final[:,:,:,0,0] = x_new[:,:,:,0,0]
    x_final[:,:,:,0,1] = x_new[:,:,:,0,1]
    x_final[:,:,:,0,2] = x_new[:,:,:,0,2]
    x_final[:,:,:,0,3] = x_new[:,:,:,0,3]

    x_final[:,:,:,1,0] = x_new[:,:,:,1,0] 
    x_final[:,:,:,1,1] = x_new[:,:,:,1,1] - x_new[:,:,:,1,3]*l*v
    x_final[:,:,:,1,2] = x_new[:,:,:,1,2]
    x_final[:,:,:,1,3] = x_new[:,:,:,1,3]

    x_final[:,:,:,2,0] = x_new[:,:,:,2,0] - l*(x_new[:,:,:,2,0]-x_xyxy[:,:,:,2,0])
    x_final[:,:,:,2,1] = x_new[:,:,:,2,1] + l*(x_xyxy[:,:,:,2,3]-x_new[:,:,:,2,1])
    x_final[:,:,:,2,2] = x_new[:,:,:,2,2]
    x_final[:,:,:,2,3] = x_new[:,:,:,2,3]

    x_final[:,:,:,3,0] = x_new[:,:,:,3,0] + l*(x_xyxy[:,:,:,3,2]-x_new[:,:,:,3,0])
    x_final[:,:,:,3,1] = x_new[:,:,:,3,1] + l*(x_xyxy[:,:,:,3,3]-x_new[:,:,:,3,1])
    x_final[:,:,:,3,2] = x_new[:,:,:,3,2]
    x_final[:,:,:,3,3] = x_new[:,:,:,3,3]

    x_final[:,:,:,4,0] = x_new[:,:,:,4,0]
    x_final[:,:,:,4,1] = x_new[:,:,:,4,1]
    x_final[:,:,:,4,2] = x_new[:,:,:,4,2]
    x_final[:,:,:,4,3] = x_new[:,:,:,4,3]

    x_final[:,:,:,5,0] = x_new[:,:,:,5,0] 
    x_final[:,:,:,5,1] = x_new[:,:,:,5,1] - x_new[:,:,:,5,3]*l*v
    x_final[:,:,:,5,2] = x_new[:,:,:,5,2]
    x_final[:,:,:,5,3] = x_new[:,:,:,5,3]

    x_final[:,:,:,6,0] = x_new[:,:,:,6,0] - l*(x_new[:,:,:,6,0]-x_xyxy[:,:,:,6,0])
    x_final[:,:,:,6,1] = x_new[:,:,:,6,1] + l*(x_xyxy[:,:,:,6,3]-x_new[:,:,:,6,1])
    x_final[:,:,:,6,2] = x_new[:,:,:,6,2]
    x_final[:,:,:,6,3] = x_new[:,:,:,6,3]

    x_final[:,:,:,7,0] = x_new[:,:,:,7,0] + l*(x_xyxy[:,:,:,7,2]-x_new[:,:,:,7,0])
    x_final[:,:,:,7,1] = x_new[:,:,:,7,1] + l*(x_xyxy[:,:,:,7,3]-x_new[:,:,:,7,1])
    x_final[:,:,:,7,2] = x_new[:,:,:,7,2]
    x_final[:,:,:,7,3] = x_new[:,:,:,7,3]

    x_final[:,:,:,8,0] = x_new[:,:,:,8,0]
    x_final[:,:,:,8,1] = x_new[:,:,:,8,1]
    x_final[:,:,:,8,2] = x_new[:,:,:,8,2]
    x_final[:,:,:,8,3] = x_new[:,:,:,8,3]

    x_final[:,:,:,9,0] = x_new[:,:,:,9,0] 
    x_final[:,:,:,9,1] = x_new[:,:,:,9,1] - x_new[:,:,:,9,3]*l*v
    x_final[:,:,:,9,2] = x_new[:,:,:,9,2]
    x_final[:,:,:,9,3] = x_new[:,:,:,9,3]

    x_final[:,:,:,10,0] = x_new[:,:,:,10,0] - l*(x_new[:,:,:,10,0]-x_xyxy[:,:,:,10,0])
    x_final[:,:,:,10,1] = x_new[:,:,:,10,1] + l*(x_xyxy[:,:,:,10,3]-x_new[:,:,:,10,1])
    x_final[:,:,:,10,2] = x_new[:,:,:,10,2]
    x_final[:,:,:,10,3] = x_new[:,:,:,10,3]

    x_final[:,:,:,11,0] = x_new[:,:,:,11,0] + l*(x_xyxy[:,:,:,11,2]-x_new[:,:,:,11,0])
    x_final[:,:,:,11,1] = x_new[:,:,:,11,1] + l*(x_xyxy[:,:,:,11,3]-x_new[:,:,:,11,1])
    x_final[:,:,:,11,2] = x_new[:,:,:,11,2]
    x_final[:,:,:,11,3] = x_new[:,:,:,11,3]

    return x_final #torch.stack(b, dim=-1)

def modify_sampling_points(sampling_location,attention_weight,scales,kernal_size=3,threshold=0.1):

    sampling_location_required_shape = list(sampling_location.shape)
    sampling_location_required_shape[4] *=  (kernal_size**2)
    sampling_locations_final = torch.full(sampling_location_required_shape,999.99).to("cuda")

    attention_weight_required_shape = list(attention_weight.shape)
    attention_weight_required_shape[4] *=  (kernal_size**2)
    attention_weight_final = torch.full(attention_weight_required_shape,0.0).to("cuda")
    attention_weight_mask = torch.full(attention_weight_required_shape,0.0).to("cuda")
    attention_weight_mask_center = torch.full(attention_weight_required_shape,0.0).to("cuda")

    n_points = 4
    n_levels = 4

    conv_locations = torch.tensor([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]).to("cuda")
    gaussian_weight = torch.tensor([1,2,1,2,4,2,1,2,1])/16.0
    gaussian_weight = gaussian_weight.to("cuda")
    
    #bring in the resolution info here
    for l in range(n_levels):
        for p in range(n_points):
            for k_ind,k in enumerate(conv_locations):
                sampling_locations_final[:,:,:,l,(p*(kernal_size**2))+k_ind,0] = sampling_location[:,:,:,l,p,0]+(k[0]*(1/scales[l][1]))
                sampling_locations_final[:,:,:,l,(p*(kernal_size**2))+k_ind,1] = sampling_location[:,:,:,l,p,1]+(k[1]*(1/scales[l][0]))

    for l in range(n_levels):
        for p in range(n_points):
            for k_ind,k in enumerate(gaussian_weight):
                attention_weight_final[:,:,:,l,(p*(kernal_size**2))+k_ind] = attention_weight[:,:,:,l,p]*k
                attention_weight_mask[:,:,:,l,(p*(kernal_size**2))+k_ind] = attention_weight[:,:,:,l,p]
                if k_ind == 4:
                    attention_weight_mask_center[:,:,:,l,(p*(kernal_size**2))+k_ind] = attention_weight[:,:,:,l,p]

    new_attention_weight_mask = torch.where(attention_weight_mask > threshold,1,0)
    new_attention_weight_final = torch.where(new_attention_weight_mask == 0,attention_weight_mask_center,attention_weight_final)
    
    return sampling_locations_final,new_attention_weight_final

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets_head = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets_left_torso = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets_right_torso = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets_upper_leg = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets_lower_leg = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights_head = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights_left_torso = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights_right_torso = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights_upper_leg = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights_lower_leg = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.query_proj_head = nn.Linear(d_model, d_model)
        self.query_proj_left_torso = nn.Linear(d_model, d_model)
        self.query_proj_right_torso = nn.Linear(d_model, d_model)
        self.query_proj_upper_leg = nn.Linear(d_model, d_model)
        self.query_proj_lower_leg = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.value_proj_head = nn.Linear(d_model, d_model)
        self.value_proj_left_torso = nn.Linear(d_model, d_model)
        self.value_proj_right_torso = nn.Linear(d_model, d_model)
        self.value_proj_upper_leg = nn.Linear(d_model, d_model)
        self.value_proj_lower_leg = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_proj_head = nn.Linear(d_model, d_model)
        self.output_proj_left_torso = nn.Linear(d_model, d_model)
        self.output_proj_right_torso = nn.Linear(d_model, d_model)
        self.output_proj_upper_leg = nn.Linear(d_model, d_model)
        self.output_proj_lower_leg = nn.Linear(d_model, d_model)
        self.parts_weights = nn.Linear(d_model,6)
        self.output_proj_concatinated = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.sampling_offsets_head.weight.data, 0.)
        constant_(self.sampling_offsets_left_torso.weight.data, 0.)
        constant_(self.sampling_offsets_right_torso.weight.data, 0.)
        constant_(self.sampling_offsets_upper_leg.weight.data, 0.)
        constant_(self.sampling_offsets_lower_leg.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            self.sampling_offsets_head.bias = nn.Parameter(grid_init.view(-1))
            self.sampling_offsets_left_torso.bias = nn.Parameter(grid_init.view(-1))
            self.sampling_offsets_right_torso.bias = nn.Parameter(grid_init.view(-1))
            self.sampling_offsets_upper_leg.bias = nn.Parameter(grid_init.view(-1))
            self.sampling_offsets_lower_leg.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights_head.weight.data, 0.)
        constant_(self.attention_weights_left_torso.weight.data, 0.)
        constant_(self.attention_weights_right_torso.weight.data, 0.)
        constant_(self.attention_weights_upper_leg.weight.data, 0.)
        constant_(self.attention_weights_lower_leg.weight.data, 0.)
        constant_(self.parts_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        constant_(self.attention_weights_head.bias.data, 0.)
        constant_(self.attention_weights_right_torso.bias.data, 0.)
        constant_(self.attention_weights_left_torso.bias.data, 0.)
        constant_(self.attention_weights_upper_leg.bias.data, 0.)
        constant_(self.attention_weights_lower_leg.bias.data, 0.)
        constant_(self.parts_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.value_proj_head.weight.data)
        constant_(self.value_proj_head.bias.data, 0.)
        xavier_uniform_(self.value_proj_left_torso.weight.data)
        constant_(self.value_proj_left_torso.bias.data, 0.)
        xavier_uniform_(self.value_proj_right_torso.weight.data)
        constant_(self.value_proj_right_torso.bias.data, 0.)
        xavier_uniform_(self.value_proj_upper_leg.weight.data)
        constant_(self.value_proj_upper_leg.bias.data, 0.)
        xavier_uniform_(self.value_proj_lower_leg.weight.data)
        constant_(self.value_proj_lower_leg.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        xavier_uniform_(self.query_proj_head.weight.data)
        constant_(self.query_proj_head.bias.data, 0.)
        xavier_uniform_(self.query_proj_left_torso.weight.data)
        constant_(self.query_proj_left_torso.bias.data, 0.)
        xavier_uniform_(self.query_proj_right_torso.weight.data)
        constant_(self.query_proj_right_torso.bias.data, 0.)
        xavier_uniform_(self.query_proj_upper_leg.weight.data)
        constant_(self.query_proj_upper_leg.bias.data, 0.)
        xavier_uniform_(self.query_proj_lower_leg.weight.data)
        constant_(self.query_proj_lower_leg.bias.data, 0.)
        xavier_uniform_(self.output_proj_head.weight.data)
        constant_(self.output_proj_head.bias.data, 0.)
        xavier_uniform_(self.output_proj_left_torso.weight.data)
        constant_(self.output_proj_left_torso.bias.data, 0.)
        xavier_uniform_(self.output_proj_right_torso.weight.data)
        constant_(self.output_proj_right_torso.bias.data, 0.)
        xavier_uniform_(self.output_proj_upper_leg.weight.data)
        constant_(self.output_proj_upper_leg.bias.data, 0.)
        xavier_uniform_(self.output_proj_lower_leg.weight.data)
        constant_(self.output_proj_lower_leg.bias.data, 0.)
        xavier_uniform_(self.output_proj_concatinated.weight.data)
        constant_(self.output_proj_concatinated.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        query_head = self.query_proj_head(query)
        query_left_torso = self.query_proj_left_torso(query)
        query_right_torso = self.query_proj_right_torso(query)
        query_upper_leg = self.query_proj_upper_leg(query)
        query_lower_leg = self.query_proj_lower_leg(query)

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        value_head = self.value_proj_head(input_flatten)
        if input_padding_mask is not None:
            value_head = value_head.masked_fill(input_padding_mask[..., None], float(0))
        value_head = value_head.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        value_left_torso = self.value_proj_left_torso(input_flatten)
        if input_padding_mask is not None:
            value_left_torso = value_left_torso.masked_fill(input_padding_mask[..., None], float(0))
        value_left_torso = value_left_torso.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        value_right_torso = self.value_proj_right_torso(input_flatten)
        if input_padding_mask is not None:
            value_right_torso = value_right_torso.masked_fill(input_padding_mask[..., None], float(0))
        value_right_torso = value_right_torso.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        value_upper_leg = self.value_proj_upper_leg(input_flatten)
        if input_padding_mask is not None:
            value_upper_leg = value_upper_leg.masked_fill(input_padding_mask[..., None], float(0))
        value_upper_leg = value_upper_leg.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        value_lower_leg = self.value_proj_lower_leg(input_flatten)
        if input_padding_mask is not None:
            value_lower_leg = value_lower_leg.masked_fill(input_padding_mask[..., None], float(0))
        value_lower_leg = value_lower_leg.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)


        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        sampling_offsets_head = self.sampling_offsets_head(query_head).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights_head = self.attention_weights_head(query_head).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights_head = F.softmax(attention_weights_head, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        sampling_offsets_left_torso = self.sampling_offsets_left_torso(query_left_torso).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights_left_torso = self.attention_weights_left_torso(query_left_torso).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights_left_torso = F.softmax(attention_weights_left_torso, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        sampling_offsets_right_torso = self.sampling_offsets_right_torso(query_right_torso).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights_right_torso = self.attention_weights_right_torso(query_right_torso).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights_right_torso = F.softmax(attention_weights_right_torso, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        sampling_offsets_upper_leg = self.sampling_offsets_upper_leg(query_upper_leg).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights_upper_leg = self.attention_weights_upper_leg(query_upper_leg).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights_upper_leg = F.softmax(attention_weights_upper_leg, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        sampling_offsets_lower_leg = self.sampling_offsets_lower_leg(query_lower_leg).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights_lower_leg = self.attention_weights_lower_leg(query_lower_leg).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights_lower_leg = F.softmax(attention_weights_lower_leg, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        parts_weights = self.parts_weights(query).view(N, Len_q, 6)
        parts_weights = F.softmax(parts_weights, -1).view(N, Len_q, 6)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
            sampling_locations_head = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets_head / offset_normalizer[None, None, None, :, None, :]
            
            sampling_locations_left_torso = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets_left_torso / offset_normalizer[None, None, None, :, None, :]
            
            sampling_locations_right_torso = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets_right_torso / offset_normalizer[None, None, None, :, None, :]
            
            sampling_locations_upper_leg = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets_upper_leg / offset_normalizer[None, None, None, :, None, :]
            
            sampling_locations_lower_leg = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets_lower_leg / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:

            #ajay's output
            #new_reference_points = box_cxcywh_to_5_point_split_v4(reference_points)
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            
            new_reference_points_head = box_cxcywh_to_head(reference_points)
            sampling_locations_head = new_reference_points_head[:, :, None, :, None, :2] \
                                 + sampling_offsets_head / self.n_points * new_reference_points_head[:, :, None, :, None, 2:] * 0.5
            
            new_reference_points_left_torso = box_cxcywh_to_left_torso(reference_points)
            sampling_locations_left_torso = new_reference_points_left_torso[:, :, None, :, None, :2] \
                                 + sampling_offsets_left_torso / self.n_points * new_reference_points_left_torso[:, :, None, :, None, 2:] * 0.5
            
            new_reference_points_right_torso = box_cxcywh_to_right_torso(reference_points)
            sampling_locations_right_torso = new_reference_points_right_torso[:, :, None, :, None, :2] \
                                 + sampling_offsets_right_torso / self.n_points * new_reference_points_right_torso[:, :, None, :, None, 2:] * 0.5
            
            new_reference_points_upper_leg = box_cxcywh_to_upper_leg(reference_points)
            sampling_locations_upper_leg = new_reference_points_upper_leg[:, :, None, :, None, :2] \
                                 + sampling_offsets_upper_leg / self.n_points * new_reference_points_upper_leg[:, :, None, :, None, 2:] * 0.5
            
            new_reference_points_lower_leg = box_cxcywh_to_lower_leg(reference_points)
            sampling_locations_lower_leg = new_reference_points_lower_leg[:, :, None, :, None, :2] \
                                 + sampling_offsets_lower_leg / self.n_points * new_reference_points_lower_leg[:, :, None, :, None, 2:] * 0.5

            #sampling_locations = reference_points[:, :, None, :, None, :2] \
            #                     + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5

            #torch.save(new_reference_points, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/new_reference_points.pt')
            torch.save(sampling_locations, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/sampling_point_tensor.pt')
            torch.save(sampling_locations_head, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/sampling_point_tensor_head.pt')
            torch.save(sampling_locations_left_torso, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/sampling_point_tensor_left_torso.pt')
            torch.save(sampling_locations_right_torso, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/sampling_point_tensor_right_torso.pt')
            torch.save(sampling_locations_upper_leg, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/sampling_point_tensor_upper_leg.pt')
            torch.save(sampling_locations_lower_leg, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/sampling_point_tensor_lower_leg.pt')

            torch.save(attention_weights, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/attention_tensor.pt')
            torch.save(attention_weights_head, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/attention_tensor_head.pt')
            torch.save(attention_weights_left_torso, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/attention_tensor_left_torso.pt')
            torch.save(attention_weights_right_torso, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/attention_tensor_right_torso.pt')
            torch.save(attention_weights_upper_leg, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/attention_tensor_upper_leg.pt')
            torch.save(attention_weights_lower_leg, '/home/ajay_sh/scratch/pedestrian_detection/DETR-Based/pedestrian_DINO/DINO/analysis_results/points_tensor/attention_tensor_lower_leg.pt')

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        #ajay version of code
        #t1 = time.time()
        #sampling_locations_new,attention_weights_new = modify_sampling_points(sampling_locations,attention_weights,input_spatial_shapes)
        #t2 = time.time()
        #print("Time taken to get padded sampling locations is ",(t2-t1))

        # for amp
        if value.dtype == torch.float16:
            # for mixed precision
            output = MSDeformAttnFunction.apply(
            value.to(torch.float32), input_spatial_shapes, input_level_start_index, sampling_locations.to(torch.float32), attention_weights, self.im2col_step)
            output = output.to(torch.float16)
            output = self.output_proj(output)
            return output

        #ajay version of code
        #t1 = time.time()
        #output = MSDeformAttnFunction.apply(
        #    value, input_spatial_shapes, input_level_start_index, sampling_locations_new, attention_weights_new, self.im2col_step)
        #t2 = time.time()
        #print("Time taken to compute cuda output is ",(t2-t1))
        #t1 = time.time()
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        
        output_head = MSDeformAttnFunction.apply(
            value_head, input_spatial_shapes, input_level_start_index, sampling_locations_head, attention_weights_head, self.im2col_step)
        
        output_left_torso = MSDeformAttnFunction.apply(
            value_left_torso, input_spatial_shapes, input_level_start_index, sampling_locations_left_torso, attention_weights_left_torso, self.im2col_step)
        
        output_right_torso = MSDeformAttnFunction.apply(
            value_right_torso, input_spatial_shapes, input_level_start_index, sampling_locations_right_torso, attention_weights_right_torso, self.im2col_step)
        
        output_upper_leg = MSDeformAttnFunction.apply(
            value_upper_leg, input_spatial_shapes, input_level_start_index, sampling_locations_upper_leg, attention_weights_upper_leg, self.im2col_step)
        
        output_lower_leg = MSDeformAttnFunction.apply(
            value_lower_leg, input_spatial_shapes, input_level_start_index, sampling_locations_lower_leg, attention_weights_lower_leg, self.im2col_step)
        
        #t2 = time.time()
        #print("Time taken to compute cuda output is ",(t2-t1))
        output = self.output_proj(output)
        output_head = self.output_proj_head(output_head)
        output_left_torso = self.output_proj_left_torso(output_left_torso)
        output_right_torso = self.output_proj_right_torso(output_right_torso)
        output_upper_leg = self.output_proj_upper_leg(output_upper_leg)
        output_lower_leg = self.output_proj_lower_leg(output_lower_leg)

        #print(output.shape)
        #print(parts_weights.shape)
        #print(parts_weights[:,:,0])
        #print(parts_weights[:,:,0][:,:,None].shape)
        #print(parts_weights[:,:,0][:,:,None].repeat(1,1,self.d_model).shape)
        #print(parts_weights[:,:,0][:,:,None].repeat(1,1,self.d_model))
        output = output*parts_weights[:,:,0][:,:,None].repeat(1,1,self.d_model)
        output_head = output_head*parts_weights[:,:,1][:,:,None].repeat(1,1,self.d_model)
        output_left_torso = output_left_torso*parts_weights[:,:,2][:,:,None].repeat(1,1,self.d_model)
        output_right_torso = output_right_torso*parts_weights[:,:,3][:,:,None].repeat(1,1,self.d_model)
        output_upper_leg = output_upper_leg*parts_weights[:,:,4][:,:,None].repeat(1,1,self.d_model)
        output_lower_leg = output_lower_leg*parts_weights[:,:,5][:,:,None].repeat(1,1,self.d_model)

        final_output = output + output_head + output_left_torso + output_right_torso + output_upper_leg + output_lower_leg
        final_output = self.output_proj_concatinated(final_output)

        return final_output
