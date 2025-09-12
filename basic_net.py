''' Incremental-Classifier Learning
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import *
from transformers import SegformerForImageClassification


#  SEGFORME Backbone
class SegFormerBackbone(torch.nn.Module):
    def __init__(self,num_classes):
        super(SegFormerBackbone, self).__init__()
        # Load the pre-trained SegFormer model for image classification
        self.model = SegformerForImageClassification.from_pretrained("nvidia/mit-b1", num_labels=num_classes,ignore_mismatched_sizes=True)
        #self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
    def forward(self, pixel_values):
        # Forward pass through the SegFormer model
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits , outputs.logits     #.logits  # The logits are the model predictions (classification).
    
    
    