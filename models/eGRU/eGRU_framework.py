import torch
import torch.nn as nn
import torch.nn.functional as F

from models.eGRU.eGRU import eGRU
from einops import rearrange, repeat



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.use_cuda = args.use_gpu
        self.input_size = args.enc_in
        self.hidden_size = args.embed_dim
        self.num_layers = args.encoder_depth
        self.batch_first = args.batch_first
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.bidirectional = False
        self.patch_size = args.patch_size
        self.GRUE = args.GRUE
        self.percentile = args.percentile
        self.patch_num = args.seq_len // self.patch_size
        args.output_attention = False
        args.d_model = args.embed_dim

        self.embed_dim = args.d_model
        self.decoder_embed_dim = args.d_model
        self.label_len = args.seq_len

        self.patch_embedding_blocks = Temp_embedding(self.patch_size, self.hidden_size)

        if self.GRUE:
            self.gru = eGRU(self.patch_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                            dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            self.gru = nn.GRU(self.patch_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              dropout=self.dropout, bidirectional=self.bidirectional)


        # Linear layer
        self.linear = nn.Linear(self.hidden_size, 1, bias=True)
        self.alpha = args.percentile/100
        self.output_layer = nn.Linear(self.decoder_embed_dim, 1, bias=True)

    def forward(self, x):
        # Divide inputs into patches
        batch_size, seq_len, _ = x.size()
        if self.GRUE:
            x_label = x[:, :, -1:]
            x = x[:, :, :-1]
            # Segmentation
            batch_size, seq_len, _ = x.size()
            x_label_segment = rearrange(x_label, 'b (seg_num seg_len) d -> b d seg_num seg_len', seg_len=self.patch_size)
            x_label_segment = torch.sum(x_label_segment, 3)
            x_label_segment = torch.gt(x_label_segment, self.patch_size*0.5)#(100-self.percentile)/100)
            x_label_segment = torch.where(x_label_segment, 1, 0)
            x_label_segment = rearrange(x_label_segment, 'b d (seg_num l_dim) -> b d seg_num l_dim', l_dim=1)
            x_label_segment = repeat(x_label_segment, 'b d seg_num l_dim -> b (repeat d) seg_num l_dim', repeat=self.input_size)

        x = rearrange(x, 'b (seg_num seg_len) d -> b d seg_num seg_len', seg_len=self.patch_size)
        x = torch.cat((x, x_label_segment), -1) if self.GRUE else x
        x = rearrange(x, 'b d seg_num d_model -> (b d) seg_num d_model')

        if self.GRUE:
            output, hidden_states, layers_out_nor, layers_out_ext = self.gru(x)
        else:
            output, hidden_states = self.gru(x)

        decoder_output = hidden_states[self.num_layers-1]
        pred = self.output_layer(decoder_output)
        pred = rearrange(pred, '(b d) pred_len -> b pred_len d', b=batch_size)
        output = pred[:, -1:, :]
        return output




