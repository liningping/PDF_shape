#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bert import BertEncoder,BertClf
from src.models.image import ImageEncoder,ImageClf


class MultimodalLateFusionClf_shape(nn.Module):
    def __init__(self, args):
        super(MultimodalLateFusionClf_shape, self).__init__()
        self.args = args

        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)
        
    def forward(self, txt, mask, segment, img, choice):
        txt_out,txt_f = self.txtclf(txt, mask, segment)
        img_out,img_f = self.imgclf(img)
        txt_img_out = 0.5*txt_out+0.5*img_out

        return txt_img_out, txt_out, img_out

