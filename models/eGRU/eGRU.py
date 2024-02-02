import torch.nn as nn
import torch
import numbers
import warnings
import math
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torch.nn.functional import linear
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt

def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)

# https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
# Pytorch GRU cell API
# Note that pytorch set batch size as second dimension by default.
class eGRU_cell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(eGRU_cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        num_chunks = 3
        self.weight_ih = Parameter(torch.empty((num_chunks * hidden_size, input_size), **factory_kwargs))
        self.weight_hh = Parameter(torch.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias_ih = Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
            self.bias_hh = Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.activation_1 = nn.Sigmoid()
        self.activation_2 = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> [Tensor, Tensor]:
        assert input.dim() in (1, 2), \
            f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx_tm1_nor = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx_tm1_ext = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx_tm1_nor = hx[:input.size(0), :]
            hx_tm1_ext = hx[input.size(0):, :]
            hx_tm1_nor = hx_tm1_nor.unsqueeze(0) if not is_batched else hx_tm1_nor
            hx_tm1_ext = hx_tm1_ext.unsqueeze(0) if not is_batched else hx_tm1_ext

        label_vec = torch.clone(torch.tensor(input[:, input.size(1)-1] == 1).unsqueeze(1))
        label_mat = label_vec.repeat(1, hx_tm1_nor.size(1))
        hx_tm1_com = torch.where(label_mat, hx_tm1_ext, hx_tm1_nor)

        if self.bias:
            z_t = self.activation_1(linear(input, self.weight_ih[:self.hidden_size, :], self.bias_ih[:self.hidden_size]) +
                                    linear(hx_tm1_com, self.weight_hh[:self.hidden_size, :],
                                           self.bias_hh[:self.hidden_size]))
            r_t = self.activation_1(linear(input, self.weight_ih[self.hidden_size: 2 * self.hidden_size, :],
                                           self.bias_ih[self.hidden_size: 2 * self.hidden_size]) +
                                    linear(hx_tm1_com, self.weight_hh[self.hidden_size: 2 * self.hidden_size, :],
                                           self.bias_hh[self.hidden_size: 2 * self.hidden_size]))
            # r*h
            hh = self.activation_2(linear(input, self.weight_ih[2 * self.hidden_size: 3 * self.hidden_size, :],
                                          self.bias_ih[2 * self.hidden_size: 3 * self.hidden_size]) +
                                   linear(r_t * hx_tm1_com, self.weight_hh[2 * self.hidden_size: 3 * self.hidden_size, :],
                                          self.bias_hh[2 * self.hidden_size: 3 * self.hidden_size]))
        else:
            z_t = self.activation_1(linear(input, self.weight_ih[:self.hidden_size, :]) +
                                    linear(hx_tm1_com, self.weight_hh[:self.hidden_size, :]))
            r_t = self.activation_1(linear(input, self.weight_ih[self.hidden_size: 2 * self.hidden_size, :]) +
                                    linear(hx_tm1_com, self.weight_hh[self.hidden_size: 2 * self.hidden_size, :]))
            # r*h
            hh = self.activation_2(linear(input, self.weight_ih[2 * self.hidden_size: 3 * self.hidden_size, :]) +
                                   linear(r_t * hx_tm1_com,
                                          self.weight_hh[2 * self.hidden_size: 3 * self.hidden_size, :]))

        h = (1-z_t)*hx_tm1_com + z_t*hh

        h_t_nor = torch.where(label_mat, hx_tm1_nor, h)
        h_t_ext = torch.where(label_mat, h, hx_tm1_ext)

        new_state = torch.cat((h_t_nor, h_t_ext), 0)
        return h, new_state

# eGRU cell Example
# rnn = eGRU_cell(10, 20)
# input = torch.randn(6, 3, 10)
# input[:3, :, 9] = 1
# input[3:, :, 9] = 0
# hx = torch.randn(6, 20)
# outputs = []
# for i in range(6):
#     print(i)
#     output, hx = rnn(input[i], hx)
#     outputs.append(output)


class eGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(eGRU, self).__init__()
        self.mode = "GRU"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional

        self.activation_1 = nn.Sigmoid()
        self.activation_2 = nn.Tanh()

        self.proj_size = 0

        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        gate_size = 3 * hidden_size

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                real_hidden_size = hidden_size
                layer_input_size = input_size if layer == 0 else real_hidden_size * num_directions

                w_ih = Parameter(torch.empty((gate_size, layer_input_size), **factory_kwargs))
                w_hh = Parameter(torch.empty((gate_size, real_hidden_size), **factory_kwargs))
                b_ih = Parameter(torch.empty(gate_size, **factory_kwargs))

                b_hh = Parameter(torch.empty(gate_size, **factory_kwargs))
                layer_params: Tuple[Tensor, ...] = ()

                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        self.flatten_parameters()

        self.reset_parameters()


    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super(eGRU, self).__setattr__(attr, value)

    def flatten_parameters(self) -> None:
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, num_weights,
                        self.input_size, rnn.get_cudnn_mode(self.mode),
                        self.hidden_size, self.proj_size, self.num_layers,
                        self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(eGRU, self)._apply(fn)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        # Flattens params (on CUDA)
        self.flatten_parameters()

        return ret

    # Initialize parameters.
    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(1) #if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)


    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(eGRU, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if 'proj_size' not in d:
            self.proj_size = 0

        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}',
                           'bias_hh_l{}{}', 'weight_hr_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    if self.proj_size > 0:
                        self._all_weights += [weights]
                        self._flat_weights_names.extend(weights)
                    else:
                        self._all_weights += [weights[:4]]
                        self._flat_weights_names.extend(weights[:4])
                else:
                    if self.proj_size > 0:
                        self._all_weights += [weights[:2]] + [weights[-1:]]
                        self._flat_weights_names.extend(weights[:2] + [weights[-1:]])
                    else:
                        self._all_weights += [weights[:2]]
                        self._flat_weights_names.extend(weights[:2])
        # get the parameter from self attribute.
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]

    @property
    def all_weights(self) -> List[List[Parameter]]:
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def _replicate_for_data_parallel(self):
        replica = super(eGRU, self)._replicate_for_data_parallel()
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        return replica

    def forward(self, input, hx=None):
        input = rearrange(input, 'b seq_len ts_d -> seq_len b ts_d') if self.batch_first else input
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 1 #if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor")
            max_batch_size = input.size(1) #if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        label = input[:, :, self.input_size:]
        input = input[:, :, :self.input_size]

        if self.bidirectional:
            input_backward = torch.flip(input, [0])
            label_backward = torch.flip(label, [0])

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size*2, self.hidden_size,
                             dtype=input.dtype, device=input.device)
            if self.bidirectional:
                hx_nor = hx[:self.num_layers, :max_batch_size, :]
                hx_ext = hx[:self.num_layers, max_batch_size:, :]
                hx_nor_backward = hx[self.num_layers:, :max_batch_size, :]
                hx_ext_backward = hx[self.num_layers:, max_batch_size:, :]
            else:
                hx_nor = hx[:, :max_batch_size, :]
                hx_ext = hx[:, max_batch_size:, :]
        else:
            num_directions = 2 if self.bidirectional else 1
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

            if self.bidirectional:
                hx_nor = hx[:self.num_layers, :max_batch_size, :]
                hx_ext = hx[:self.num_layers, max_batch_size:, :]
                hx_nor_backward = hx[self.num_layers:, :max_batch_size, :]
                hx_ext_backward = hx[self.num_layers:, max_batch_size:, :]
            else:
                hx_nor = hx[:, :max_batch_size, :]
                hx_ext = hx[:, max_batch_size:, :]

        label_mat = label.repeat(1, 1, self.hidden_size)
        label_mat = torch.gt(label_mat, 0)

        if self.bidirectional:
            label_mat_backward = label_backward.repeat(1, 1, hx_nor.size(2))
            label_mat_backward = torch.gt(label_mat_backward, 0)

        self.check_forward_args(input, torch.zeros(self.num_layers * num_directions,
                                                   max_batch_size, self.hidden_size,
                                                   dtype=input.dtype, device=input.device), batch_sizes)

        seq_length = input.size(0) #if self.batch_first else input.size(0)
        if batch_sizes is None:
            layers_out = torch.empty((self.num_layers, max_batch_size, self.hidden_size), dtype=input.dtype, device=input.device)
            layers_out_nor = torch.empty((self.num_layers, max_batch_size, self.hidden_size), dtype=input.dtype,
                                     device=input.device)
            layers_out_ext = torch.empty((self.num_layers, max_batch_size, self.hidden_size), dtype=input.dtype,
                                     device=input.device)
            if self.bidirectional:
                layers_out_backward = torch.empty((self.num_layers, max_batch_size, self.hidden_size), dtype=input.dtype, device=input.device)
            for layer_idx in range(self.num_layers):
                input_current_layer = input if layer_idx == 0 else hidden_states_previous_layer
                hidden_states_previous_layer = torch.empty((seq_length, max_batch_size, self.hidden_size), dtype=input.dtype, device=input.device)

                hx_current_layer_nor = "Will NOT be referenced before assigned as the hidden states at the end of first time step"
                hx_current_layer_ext = "Will NOT be referenced before assigned as the hidden states at the end of first time step"
                for time_step_idx in range(seq_length):
                    tmp_label_mat_layer_time_step = label_mat[time_step_idx]
                    hx_current_layer_tm1_nor = hx_nor[layer_idx, :, :] if time_step_idx == 0 else hx_current_layer_nor
                    hx_current_layer_tm1_ext = hx_ext[layer_idx, :, :] if time_step_idx == 0 else hx_current_layer_ext

                    hx_current_layer_tm1 = torch.where(tmp_label_mat_layer_time_step, hx_current_layer_tm1_ext, hx_current_layer_tm1_nor)

                    para_idx_forward = self._flat_weights_names.index('weight_ih_l{}'.format(layer_idx))
                    tmp_weight_ih = self._flat_weights[para_idx_forward]
                    tmp_weight_hh = self._flat_weights[para_idx_forward+1]
                    tmp_bias_ih = self._flat_weights[para_idx_forward+2]
                    tmp_bias_hh = self._flat_weights[para_idx_forward+3]

                    z_t = self.activation_1(linear(input_current_layer[time_step_idx],
                                                   tmp_weight_ih[:self.hidden_size, :],
                                                   tmp_bias_ih[:self.hidden_size]) +
                                            linear(hx_current_layer_tm1,
                                                   tmp_weight_hh[:self.hidden_size, :],
                                                   tmp_bias_hh[:self.hidden_size]))
                    r_t = self.activation_1(linear(input_current_layer[time_step_idx],
                                                   tmp_weight_ih[self.hidden_size: 2 * self.hidden_size, :],
                                                   tmp_bias_ih[self.hidden_size: 2 * self.hidden_size]) +
                                            linear(hx_current_layer_tm1,
                                                   tmp_weight_hh[self.hidden_size: 2 * self.hidden_size, :],
                                                   tmp_bias_hh[self.hidden_size: 2 * self.hidden_size]))
                    # r*h
                    hh = self.activation_2(linear(input_current_layer[time_step_idx],
                                                  tmp_weight_ih[2 * self.hidden_size: 3 * self.hidden_size, :],
                                                  tmp_bias_ih[2 * self.hidden_size: 3 * self.hidden_size]) +
                                           linear(r_t * hx_current_layer_tm1,
                                                  tmp_weight_hh[2 * self.hidden_size: 3 * self.hidden_size, :],
                                                  tmp_bias_hh[2 * self.hidden_size: 3 * self.hidden_size]))

                    # Tensorflow: h = z * h_tm1 + (1 - z) * hh
                    h = (1 - z_t) * hx_current_layer_tm1 + z_t * hh
                    hx_current_layer_nor = torch.where(tmp_label_mat_layer_time_step, hx_current_layer_tm1_nor, h)
                    hx_current_layer_ext = torch.where(tmp_label_mat_layer_time_step, h, hx_current_layer_tm1_ext)

                    hidden_states_previous_layer[time_step_idx] = h
                layers_out[layer_idx] = h
                layers_out_nor[layer_idx] = hx_current_layer_nor
                layers_out_ext[layer_idx] = hx_current_layer_ext

                # Backward
                if self.bidirectional is True:
                    # Backward
                    input_current_layer_backward = input_backward if layer_idx == 0 else hidden_states_previous_layer_backward
                    hidden_states_previous_layer_backward = torch.empty((seq_length, max_batch_size, self.hidden_size),
                                                               dtype=input.dtype, device=input.device)

                    hx_current_layer_nor_backward = "Will NOT be referenced before assigned as the hidden states at the end of first time step"
                    hx_current_layer_ext_backward = "Will NOT be referenced before assigned as the hidden states at the end of first time step"
                    for time_step_idx in range(seq_length):
                        hx_current_layer_tm1_nor_backward = hx_nor_backward[layer_idx, :, :] if time_step_idx == 0 else hx_current_layer_nor_backward
                        hx_current_layer_tm1_ext_backward = hx_ext_backward[layer_idx, :, :] if time_step_idx == 0 else hx_current_layer_ext_backward
                        hx_current_layer_tm1_backward = torch.where(label_mat_backward[time_step_idx], hx_current_layer_tm1_ext_backward,
                                                           hx_current_layer_tm1_nor_backward)

                        para_idx_backward = self._flat_weights_names.index('weight_ih_l{}_reverse'.format(layer_idx))
                        tmp_weight_ih_backward = self._flat_weights[para_idx_backward]
                        tmp_weight_hh_backward = self._flat_weights[para_idx_backward + 1]
                        tmp_bias_ih_backward = self._flat_weights[para_idx_backward + 2]
                        tmp_bias_hh_backward = self._flat_weights[para_idx_backward + 3]

                        z_t_backward = self.activation_1(linear(input_current_layer_backward[time_step_idx],
                                                       tmp_weight_ih_backward[:self.hidden_size, :],
                                                       tmp_bias_ih_backward[:self.hidden_size]) +
                                                linear(hx_current_layer_tm1_backward,
                                                       tmp_weight_hh_backward[:self.hidden_size, :],
                                                       tmp_bias_hh_backward[:self.hidden_size]))
                        r_t_backward = self.activation_1(linear(input_current_layer_backward[time_step_idx],
                                                       tmp_weight_ih_backward[self.hidden_size: 2 * self.hidden_size, :],
                                                       tmp_bias_ih_backward[self.hidden_size: 2 * self.hidden_size]) +
                                                linear(hx_current_layer_tm1_backward,
                                                       tmp_weight_hh_backward[self.hidden_size: 2 * self.hidden_size, :],
                                                       tmp_bias_hh_backward[self.hidden_size: 2 * self.hidden_size]))
                        # r*h
                        hh_backward = self.activation_2(linear(input_current_layer_backward[time_step_idx],
                                                      tmp_weight_ih_backward[2 * self.hidden_size: 3 * self.hidden_size, :],
                                                      tmp_bias_ih_backward[2 * self.hidden_size: 3 * self.hidden_size]) +
                                               linear(r_t_backward * hx_current_layer_tm1_backward,
                                                      tmp_weight_hh_backward[2 * self.hidden_size: 3 * self.hidden_size, :],
                                                      tmp_bias_hh_backward[2 * self.hidden_size: 3 * self.hidden_size]))

                        h_backward = (1 - z_t_backward) * hx_current_layer_tm1_backward + z_t_backward * hh_backward
                        hx_current_layer_nor_backward = torch.where(label_mat_backward[time_step_idx], hx_current_layer_tm1_nor_backward, h_backward)
                        hx_current_layer_ext_backward = torch.where(label_mat_backward[time_step_idx], h_backward, hx_current_layer_tm1_ext_backward)

                        hidden_states_previous_layer_backward[time_step_idx] = h_backward
                    layers_out_backward[layer_idx] = h_backward

                    hidden_states_previous_layer = torch.cat((hidden_states_previous_layer, torch.flip(hidden_states_previous_layer_backward, [0])), 2)
                    hidden_states_previous_layer_backward = torch.flip(hidden_states_previous_layer, [0])

                if self.dropout:
                    if self.num_layers == 1:
                        warnings.warn("UserWarning: dropout option adds dropout after all but last recurrent layer, "
                                      "so non-zero dropout expects num_layers greater than 1, "
                                      "but got dropout={:5.4f} and num_layers=()".format(self.dropout, self.num_layers))
                    if layer_idx < self.num_layers -1:
                        hidden_states_previous_layer = F.dropout(hidden_states_previous_layer, self.dropout, training=self.training)
                        layers_out[layer_idx] = F.dropout(layers_out[layer_idx], self.dropout, training=self.training)
                        if self.bidirectional:
                            hidden_states_previous_layer_backward = F.dropout(hidden_states_previous_layer_backward, self.dropout, training=self.training)
                            layers_out_backward[layer_idx] = F.dropout(layers_out_backward[layer_idx], self.dropout, training=self.training)

            if self.bidirectional:
                layers_out = torch.cat((layers_out, layers_out_backward), 0)
            result = tuple([hidden_states_previous_layer, layers_out])

        else:
            result = 0
            # NOT IMPLEMENTED YET

        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = hidden.squeeze(1)

            output = rearrange(output, 'seq_len b ts_d -> b seq_len ts_d') if self.batch_first else output
            return output, self.permute_hidden(hidden, unsorted_indices), layers_out_nor, layers_out_ext


# eGRU Layer Example
# rnn = eGRU(10, 20, 2)
# input = torch.randn(5, 3, 11)
# input[:, :2, 10] = 1
# input[:, 2:, 10] = 0
# h0 = torch.randn(2, 6, 20)
# output, hidden_states, layers_out_nor, layers_out_ext = rnn(input)



