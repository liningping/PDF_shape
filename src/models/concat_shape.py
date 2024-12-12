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

from .bert import BertEncoder, BertClf
from .image import ImageEncoder, ImageClf


class MultimodalConcatBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBertClf, self).__init__()
        self.args = args
        self.txtenc = BertEncoder(args)
        self.imgenc = ImageEncoder(args)
        self.txtclf = BertClf(args)
        self.imgclf = ImageClf(args)

        last_size = args.hidden_sz + (args.img_hidden_sz * args.num_image_embeds)
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            if args.include_bn:
                self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden

        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, txt, mask, segment, img):
        txt_clf,txt_embed = self.txtclf(txt, mask, segment)
        img_clf,img_embed = self.imgclf(img)
        img_embed = torch.flatten(img_embed, start_dim=1)
        out = torch.cat([txt_embed, img_embed], -1)
        for layer in self.clf:
            final_out = layer(final_out)
        return final_out, txt_clf, img_clf
