"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
from utils import Node


def _construct_dags(prev_nodes, activations, func_names, num_blocks, layer_sizes=None):
    """Constructs a set of DAGs based on the actions, i.e., previous nodes and
    activation functions, sampled from the controller/policy pi.

    Args:
        prev_nodes: Previous node actions from the policy.
        activations: Activations sampled from the policy.
        func_names: Mapping from activation function names to functions.
        num_blocks: Number of blocks in the target RNN cell.
        layer_sizes: Number of neurons at each layer (if available)

    Returns:
        A list of DAGs defined by the inputs.

    RNN cell DAGs are represented in the following way:

    1. Each element (node) in a DAG is a list of `Node`s.

    2. The `Node`s in the list dag[i] correspond to the subsequent nodes
       that take the output from node i as their own input.

    3. dag[-1] is the node that takes input from x^{(t)} and h^{(t - 1)}.
       dag[-1] always feeds dag[0].
       dag[-1] acts as if `w_xc`, `w_hc`, `w_xh` and `w_hh` are its
       weights.

    4. dag[N - 1] is the node that produces the hidden state passed to
       the next timestep. dag[N - 1] is also always a leaf node, and therefore
       is always averaged with the other leaf nodes and fed to the output
       decoder.
    """
    dags = []
    for idx, (nodes, func_ids) in enumerate(zip(prev_nodes, activations)):

        if layer_sizes is not None:
            block_layer_sizes = layer_sizes[idx]
        else:
            block_layer_sizes = [""] * num_blocks

        dag = collections.defaultdict(list)

        # add first node
        dag[-1] = [Node(0, func_names[func_ids[0]] + "\n size: {}".format(block_layer_sizes[0]))]
        dag[-2] = [Node(0, func_names[func_ids[0]] + "\n size: {}".format(block_layer_sizes[0]))]

        # add following nodes
        for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
            dag[utils.to_item(idx)].append(Node(jdx + 1, func_names[func_id] + "\n size: {}".format(block_layer_sizes[idx+1])))

        leaf_nodes = set(range(num_blocks)) - dag.keys()

        # merge with avg
        for idx in leaf_nodes:
            dag[idx] = [Node(num_blocks, 'avg')]

        # TODO(brendan): This is actually y^{(t)}. h^{(t)} is node N - 1 in
        # the graph, where N Is the number of nodes. I.e., h^{(t)} takes
        # only one other node as its input.
        # last h[t] node
        last_node = Node(num_blocks + 1, 'h[t]')
        dag[num_blocks] = [last_node]
        dags.append(dag)

    return dags


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
     language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        network_types = self.args.network_type.split('_')
        for network_type in network_types:

            if network_type == 'rnn':
                # NOTE(brendan): `num_tokens` here is just the activation function
                # for every even step,
                self.num_tokens = [len(args.shared_rnn_activations)]
                for idx in range(self.args.num_blocks):
                    self.num_tokens += [idx + 1,
                                        len(args.shared_rnn_activations)]
                self.func_names = args.shared_rnn_activations
            elif network_type == 'cnn':
                self.num_tokens = [len(args.shared_cnn_types),
                                   self.args.num_blocks]
                self.func_names = args.shared_cnn_types
            elif network_type == 'mlp':
                self.num_mlp_tokens = [len(args.shared_mlp_activations)]
                for idx in range(self.args.num_mlp_blocks):
                    self.num_mlp_tokens += [idx + 1,
                                        len(args.shared_rnn_activations)]


        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        #  shared? At least for the activation functions, which all have the
        #  same size.
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        # mlp parameters
        num_total_mlp_tokens = sum(self.num_mlp_tokens)
        self.mlp_encoder = torch.nn.Embedding(num_total_mlp_tokens, args.controller_hid)
        self.mlp_lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
        self.mlp_inputs_enc = torch.nn.Embedding(3, args.controller_hid)

        self.layer_range = self.args.max_layer_size - self.args.min_layer_size + 1

        self.mlp_decoders = {'input': torch.nn.Linear(args.controller_hid, args.controller_hid),
                             'residual': torch.nn.Linear(args.controller_hid, args.controller_hid),
                             'activation': torch.nn.Linear(args.controller_hid, len(args.shared_rnn_activations)),
                             'layer_size': torch.nn.Linear(args.controller_hid, 1),
                             'input_selection': torch.nn.Linear(args.controller_hid, self.layer_range)}

        self.mlp_embs = {'inputs': torch.nn.Embedding(3, args.controller_hid),
                         'layer_sizes': torch.nn.Embedding(self.layer_range, args.controller_hid),
                         'activations': torch.nn.Embedding(len(self.args.shared_rnn_activations), args.controller_hid)}


        self._decoders = torch.nn.ModuleList(self.decoders + list(self.mlp_decoders.values()))

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def mlp_forward(self, inputs, hidden, is_embed):
        if is_embed:
            inputs = self.mlp_encoder(inputs)

        hx, cx = self.mlp_lstm(inputs, hidden)
        input_proj = self.mlp_decoders['input'](hx) / self.args.softmax_temperature
        residual_proj = self.mlp_decoders['residual'](hx) / self.args.softmax_temperature
        activation_proj = self.mlp_decoders['activation'](hx) / self.args.softmax_temperature
        layer_size_proj = self.mlp_decoders['layer_size'](hx) / self.args.softmax_temperature

        projs = {'input': input_proj,
                 'residual': residual_proj,
                 'activation': activation_proj,
                 'layer_size': layer_size_proj}

        return projs, (hx, cx)


    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        for block_idx in range(2*(self.args.num_blocks - 1) + 1):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
            # .view()? Same below with `action`.
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 2
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:mode]),
                requires_grad=False)

            if mode == 0:
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        dags = _construct_dags(prev_nodes,
                               activations,
                               self.func_names,
                               self.args.num_blocks)

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def sample_mlp(self, batch_size=1, save_dir=None, with_details=False):
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        # network_res = []
        network_inps = []
        network_act = []
        network_log_probs = []
        network_entropies = []
        network_sizes = []
        # There are 3 possible inputs in this function. Account for each
        all_layers = [self.mlp_inputs_enc(torch.tensor([i])) for i in range(3)]
        all_layers_proj = [self.mlp_decoders['input'](self.mlp_inputs_enc(torch.tensor([i]))) for i in range(3)]

        for block_idx in range(self.mlp_layers):
            # Determine Network Input
            inputs, hidden = self.mlp_forward(inputs=inputs, hidden=hidden)
            input_proj = self.mlp_decoders['input'](inputs) / self.args.softmax_temperature
            all_layers.append(inputs)
            all_layers_proj.append(input_proj)
            # [NumPrevLayers, HiddenSize]
            possible_inputs = torch.cat(all_layers_proj[:-1], dim=0)
            possible_input_logits = self.mlp_decoders['input_selection'](possible_inputs)
            possible_input_logits = torch.reshape(possible_input_logits, [1, -1])

            possible_input_probs = F.softmax(possible_input_logits, dim=-1)
            possible_input_log_probs = F.log_softmax(possible_input_logits, dim=-1)

            selected_input = possible_input_probs.multinomial(num_samples=1).data
            selected_log_prob = possible_input_log_probs.gather(1, utils.get_variable(selected_input,
                                                                                      requires_grad=False))
            entropy = -(possible_input_log_probs * possible_input_probs).sum(1, keepdim=False)
            # TODO: Check this
            network_entropies.append(entropy)
            network_log_probs.append(selected_log_prob[:, 0])
            network_inps.append(selected_input[:, 0])

            inputs = possible_inputs[selected_input]

            # Determine Layer Size
            inputs, hidden = self.mlp_forward(inputs=inputs, hidden=hidden)
            layer_size_logits = self.mlp_decoders['layer_size'](inputs)
            layer_size_probs = F.softmax(layer_size_logits, dim=-1)
            layer_size_log_probs = F.log_softmax(layer_size_logits, dim=-1)
            selected_layer_size = layer_size_probs.multinomial(num_samples=1).data
            selected_layer_size_log_prob = layer_size_log_probs.gather(1,
                                                                       utils.get_variable(selected_layer_size,
                                                                                          requires_grad=False))
            entropy = -(layer_size_log_probs * layer_size_probs).sum(1, keepdim=False)
            network_entropies.append(entropy)
            network_log_probs.append(selected_layer_size_log_prob)

            network_sizes.append(selected_layer_size + self.args.min_layer_size)
            inputs = self.mlp_embs['layer_sizes'](selected_layer_size)

            # Determine Activation
            inputs, hidden = self.mlp_forward(inputs=inputs, hidden=hidden)
            activation_logits = self.mlp_decoders['activation'](inputs)
            activation_probs = F.softmax(activation_logits, dim=-1)
            activation_log_probs = F.log_softmax(activation_logits, dim=-1)
            selected_activation = activation_probs.multinomial(num_samples=1).data
            selected_activation_log_prob = activation_log_probs.gather(
                1,
                utils.get_variable(activation_log_probs, requires_grad=False))
            entropy = -(activation_log_probs * activation_probs).sum(1, keepdim=False)

            network_entropies.append(entropy)
            network_log_probs.append(selected_activation_log_prob)
            network_act.append(selected_activation)

            inputs = self.mlp_embs['activations'](selected_activation)

        prev_nodes = torch.stack(network_inps).transpose(0, 1)
        activations = torch.stack(network_act).transpose(0, 1)

        dags = _construct_dags(prev_nodes,
                               activations,
                               self.func_names,
                               self.args.num_mlp_blocks,
                               layer_sizes=network_sizes)

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            return dags, torch.cat(network_log_probs), torch.cat(network_entropies)

        return dags


    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))

