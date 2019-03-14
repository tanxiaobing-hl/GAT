# coding=utf-8

import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

'''
整个GAT（Graph Attention Network）分为多层：
   1. 每一层都是一个多头注意力机制实现的网络
   2. 层数由list n_heads的元素个数指定，每一层注意力机制的头数由元素的值指定，中间的被称为隐藏层
'''
class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        
        # 第一层, 多头之间使用concatenation
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        
        # 中间层, 多头之间使用concatenation
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        
        # 最后一层，正如论文所述，这一层没有采用concatenation，而是采用average
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits
