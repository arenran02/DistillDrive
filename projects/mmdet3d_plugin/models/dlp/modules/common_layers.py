import copy
import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    def __init__(self, 
        d_model=256, 
        nhead=8,
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dim_feedforward=1024, 
        dropout=0.1,
        activation="relu", 
        return_intermediate_dec=False,
        extra_track_attn=False, 
        n_detect_query=50):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(
            d_model,
            dim_feedforward, 
            dropout, 
            activation, 
            nhead)

        self.encoder = TransformerEncoder(
            encoder_layer, 
            num_encoder_layers)

        if num_decoder_layers is not None:
            decoder_layer = TransformerDecoderLayer(
                d_model, 
                dim_feedforward, 
                dropout, 
                activation,
                nhead, 
                n_detect_query=n_detect_query, 
                extra_track_attn=extra_track_attn)

            self.decoder = TransformerDecoder(
                decoder_layer, 
                num_decoder_layers,
                return_intermediate_dec)
        else:
            self.decoder = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None, src_padding_mask=None, tgt_padding_mask=None):
        if self.encoder is not None:
            memory = self.encoder(src, src_padding_mask)
        else:
            memory = src

        if self.decoder is not None:
            hidden_state = self.decoder(tgt=tgt, src=memory, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        else:
            hidden_state = memory

        return hidden_state


class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
        d_model=256, 
        d_ffn=1024, 
        dropout=0.1, 
        activation="relu", 
        n_heads=8):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, mask=None):
        '''
        Input:
            src: [N, T, D], 
            mask: [N, T],
        Return:
            src: [N, T, D]
        '''
        src2 = self.self_attn(src.transpose(0, 1), src.transpose(0, 1), src.transpose(0, 1), key_padding_mask=mask)[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask=None):
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(output, mask=mask)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8, n_detect_query=50, 
                 extra_track_attn=False):
        super().__init__()
        self.num_head = n_heads
        self.n_detect_query = n_detect_query

        # cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, tgt_padding_mask, print_flag=False):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(tgt, query_pos, tgt_padding_mask, print_flag)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), 
                              key_padding_mask=tgt_padding_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def _forward_track_attn(self, tgt, query_pos, tgt_padding_mask=None, print_flag=False):

        if tgt.shape[1] > self.n_detect_query:
            # tgt_padding_mask第n_detect_query列为False的行选出来
            have_track = ~tgt_padding_mask[:, self.n_detect_query]
            q = k = self.with_pos_embed(tgt, query_pos)[have_track]
            v = tgt[have_track]
            tgt2 = self.update_attn(q[:, self.n_detect_query:].transpose(0, 1),
                                    k[:, self.n_detect_query:].transpose(0, 1),
                                    v[:, self.n_detect_query:].transpose(0, 1), 
                                    key_padding_mask=tgt_padding_mask[have_track][:, self.n_detect_query:])[0].transpose(0, 1)
            tgt[have_track] = torch.cat([v[:, :self.n_detect_query], self.norm4(v[:, self.n_detect_query:] + self.dropout5(tgt2))], dim=1)
        return tgt


    def forward(self, tgt, query_pos, src, src_padding_mask=None, tgt_padding_mask=None, print_flag=False):
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, tgt_padding_mask, print_flag)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos).transpose(0, 1), 
                               src.transpose(0, 1), src.transpose(0, 1),
                               key_padding_mask=src_padding_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, query_pos=None, src_padding_mask=None, tgt_padding_mask=None):

        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, query_pos, src, src_padding_mask, tgt_padding_mask, print_flag=False)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

def build_mlp(c_in, channels, norm=None, activation="relu"):
    layers = []
    num_layers = len(channels)
    if norm is not None:
        norm = get_norm(norm)

    activation = get_activation(activation)

    for k in range(num_layers):
        if k == num_layers - 1:
            layers.append(nn.Linear(c_in, channels[k], bias=True))
        else:
            if norm is None:
                layers.extend([nn.Linear(c_in, channels[k], bias=True), activation()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, channels[k], bias=False),
                        norm(channels[k]),
                        activation(),
                    ]
                )
            c_in = channels[k]

    return nn.Sequential(*layers)


def get_norm(norm: str):
    if norm == "bn":
        return nn.BatchNorm1d
    elif norm == "ln":
        return nn.LayerNorm
    else:
        raise NotImplementedError


def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    else:
        raise NotImplementedError

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x, flag=False):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



