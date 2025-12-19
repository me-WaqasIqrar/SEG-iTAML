import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import *
from transformers import SegformerForImageClassification, SegformerForSemanticSegmentation, SegformerConfig

class SegformerBackbone(nn.Module):
    
    def __init__(self, num_class, model_name="nvidia/mit-b2", pretrained=True, device=None):
        super(SegformerBackbone, self).__init__()
        self.device = device
        
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = num_class
        if pretrained:
            self.model = SegformerForImageClassification.from_pretrained(model_name, num_labels=num_class, ignore_mismatched_sizes=True)
        else:
            self.model = SegformerForImageClassification(config)

        clf = getattr(self.model, "classifier", None)
        inferred_dim = None
        if clf is not None:
            if hasattr(clf, "in_features"):
                inferred_dim = clf.in_features
            else:
                # try first linear inside a Sequential
                try:
                    for module in clf.modules():
                        if isinstance(module, nn.Linear):
                            inferred_dim = module.in_features
                            break
                except Exception:
                    inferred_dim = None

        if inferred_dim is None:
            inferred_dim = getattr(self.model.config, "hidden_size", None)
            if inferred_dim is None:
                inferred_dim = num_class

        self.out_dim = inferred_dim

        
        if self.device is not None:
            self.to(self.device)

    def forward(self, pixel_values):
        
        # HF SegformerForImageClassification expects keyword 'pixel_values' or positional first arg
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        logits = outputs.logits  # shape (B, num_labels)
        
        # Extract features from hidden states for a different representation
        feats = self.extract(pixel_values)  # Extract actual features instead of clone
        
        # Return logits and features (two different outputs for different purposes)
        return logits, feats

    def extract(self, pixel_values):
        
        out = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            # try last hidden state: (B, seq_len, hidden_size) -> pool/mean to vector
            last_h = out.hidden_states[-1]  # (B, seq_len, hidden_size)
            feats = last_h.mean(dim=1)
        return feats

    @property
    def device(self):
        return next(self.model.parameters()).device

    @device.setter
    def device(self, value):
        if value is not None:
            self.model.to(value)

class BasicNet1(nn.Module):

    def __init__(
        self, args, use_bias=False, init="kaiming", use_multi_fc=False, device=None
    ):
        super(BasicNet1, self).__init__()

        self.use_bias = use_bias
        self.init = init
        self.use_multi_fc = use_multi_fc
        self.args = args

        if(self.args.dataset=="mnist"):
            self.convnet = RPS_net_mlp()
        elif(self.args.dataset=="svhn"):
            self.convnet = RPS_net(self.args.num_class)
        elif(self.args.dataset=="cifar100"):
            self.convnet = RPS_net(self.args.num_class)
        elif(self.args.dataset=="omniglot"):
            self.convnet = RPS_net(self.args.num_class)
        elif(self.args.dataset=="celeb"): 
            self.convnet = resnet18()
        
        elif self.args.dataset == "custom":
            # use Segformer as backbone for either image classification or semantic segmentation
            if getattr(self.args, 'segmentation', False):
                
                # segmentation model: include background label (0) + class labels (1..N)
                num_labels = self.args.num_class + 1
                class SegformerSegBackbone(nn.Module):
                    def __init__(self, num_labels, model_name="nvidia/mit-b2", pretrained=True, device=None):
                        super().__init__()
                        self.device = device
                        config = SegformerConfig.from_pretrained(model_name)
                        config.num_labels = num_labels
                        if pretrained:
                            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
                        else:
                            self.model = SegformerForSemanticSegmentation(config)

                        # output channels
                        self.out_dim = num_labels
                        if self.device is not None:
                            self.to(self.device)

                    def forward(self, pixel_values, labels=None):
                        # return logits (B, num_labels, H, W) and None for features
                        outputs = self.model(pixel_values=pixel_values, labels=labels, return_dict=True)
                        # when labels provided HF model returns loss too; we return logits so learner can compute loss if desired
                        logits = outputs.logits
                        # Segformer outputs at 1/4 resolution so i do upsample to match input size
                        
                        # pixel_values shape: (B, C, H, W), logits shape: (B, num_labels, H/4, W/4)
                        if logits.shape[-2:] != pixel_values.shape[-2:]:
                            logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False)
                        return logits, None

                self.convnet = SegformerSegBackbone(num_labels, model_name="nvidia/mit-b2", pretrained=True, device=device)
            else:
                # use Segformer as backbone for image classification
                self.convnet = SegformerBackbone(self.args.num_class, model_name="nvidia/mit-b2", pretrained=True, device=device)
        
        self.classifier = None

        self.n_classes = 0
        self._device = device
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)
        
    def forward(self, x):
        x1, x2 = self.convnet(x)
        return x1, x2

    @property
    def features_dim(self):
        return self.convnet.out_dim

    def extract(self, x):
        return self.convnet(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def copy(self):
        return copy.deepcopy(self)
    
    def add_classes(self, n_classes):
        if self.use_multi_fc:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes
        
    def _add_classes_multi_fc(self, n_classes):
        if self.classifier is None:
            self.classifier = []

        new_classifier = self._gen_classifier(n_classes)
        name = "_clf_{}".format(len(self.classifier))
        self.__setattr__(name, new_classifier)
        self.classifier.append(name)

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.n_classes + n_classes)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, n_classes):
#         torch.manual_seed(self.seed)
        classifier = nn.Linear(self.convnet.out_dim, n_classes, bias=self.use_bias).cuda()
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.)

        return classifier

    
    
    