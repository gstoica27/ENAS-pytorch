"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
from utils import Node
import numpy as np

def _construct_dags(prev_nodes, activations, func_names, num_blocks, is_rnn=True):
    """Constructs a set of DAGs based on the actions, i.e., previous nodes and
    activation functions, sampled from the controller/policy pi.

    Args:
        prev_nodes: Previous node actions from the policy.
        activations: Activations sampled from the policy.
        func_names: Mapping from activation function names to functions.
        num_blocks: Number of blocks in the target RNN cell.
        layer_info: Layer Information sampled from the policy
            - sizes: layer sizes
            - map: layer idx to size map

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
    # layer_sizes = layer_info['sizes']
    # layer_map = layer_info['map']
    dags = []
    for nodes, func_ids in zip(prev_nodes, activations):
        dag = collections.defaultdict(list)

        # add first node
        if is_rnn:
            dag[-1] = [Node(0, func_names[func_ids[0]])]
            dag[-2] = [Node(0, func_names[func_ids[0]])]
        else:
            dag[-1] = [Node(0, func_names[func_ids[0]])]

        if len(nodes) == len(func_ids):
            nodes = nodes[1:]

        # add following nodes
        for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
            dag[utils.to_item(idx)].append(Node(jdx + 1, func_names[func_id]))

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

def _construct_mlp_dags(prev_nodes, activations, func_names, num_blocks):
    """Constructs a set of DAGs based on the actions, i.e., previous nodes and
    activation functions, sampled from the controller/policy pi.

    Args:
        prev_nodes: Previous node actions from the policy.
        activations: Activations sampled from the policy.
        func_names: Mapping from activation function names to functions.
        num_blocks: Number of blocks in the target RNN cell.
        layer_info: Layer Information sampled from the policy
            - sizes: layer sizes
            - map: layer idx to size map

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
    # layer_sizes = layer_info['sizes']
    # layer_map = layer_info['map']
    dags = []
    # [Blocks, NumSamples], [Blocks]
    for nodes, func_ids in zip(prev_nodes, activations):
        dag = collections.defaultdict(list)

        # jdx, (NumSamples, Block)
        for jdx, (idxs, func_id) in enumerate(zip(nodes, func_ids)):
            # Sample
            seen_idxs = []
            for idx in idxs:
                # only add unique connections
                if idx not in seen_idxs:
                    dag[utils.to_item(idx)].append(Node(jdx+3, func_names[func_id]))
                    seen_idxs.append(idx)
        leaf_nodes = set(range(num_blocks+3)) - dag.keys()

        for idx in leaf_nodes:
            dag[idx] = [Node(num_blocks+3, 'avg')]

        last_node = Node(num_blocks+4, 'output')
        dag[num_blocks+3] = [last_node]
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
        # `num_tokens` here is just the activation function
        # for every even step,

        # if self.args.network_type == 'rnn':
        #     # NOTE(brendan): `num_tokens` here is just the activation function
        #     # for every even step,
        #     self.num_tokens = [len(args.shared_rnn_activations)]
        #     for idx in range(self.args.num_blocks):
        #         self.num_tokens += [idx + 1,
        #                             len(args.shared_rnn_activations)]
        #     self.func_names = args.shared_rnn_activations
        # elif self.args.network_type == 'cnn':
        #     self.num_tokens = [len(args.shared_cnn_types),
        #                        self.args.num_blocks]
        #     self.func_names = args.shared_cnn_types

        self.network_configs = {}
        for network_type in self.args.network_types:
            if network_type == 'rnn':
                num_tokens = [len(args.shared_rnn_activations)]
                for idx in range(self.args.num_rnn_blocks):
                    num_tokens += [idx + 1, len(args.shared_rnn_activations)]

                self.encoder = torch.nn.Embedding(sum(num_tokens), args.controller_hid)
                self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
                # TODO: These should be shared. However in the base code they are not.
                #  Experiment how sharing them affects model
                decoders = []
                for idx, size in enumerate(num_tokens):
                    decoder = torch.nn.Linear(args.controller_hid, size)
                    decoders.append(decoder)

                self._decoders = torch.nn.ModuleList(decoders)

                self.network_configs['rnn'] = {'num_tokens': num_tokens,
                                               'func_names': args.shared_rnn_activations,
                                               'encoder': self.encoder,
                                               'lstm': self.lstm,
                                               'decoders': decoders}

            elif network_type == 'mlp':
                # input embeddings for index
                self.input_embs = torch.nn.Embedding(args.num_mlp_inputs, args.controller_hid)
                # index calculation attentions
                self.idx_l_1 = torch.nn.Linear(args.controller_hid, args.controller_hid)
                self.idx_l_2 = torch.nn.Linear(args.controller_hid, args.controller_hid)
                self.idx_l_3 = torch.nn.Linear(args.controller_hid, 1)
                # layer embeddings
                self.layer_embs = torch.nn.Embedding(len(args.mlp_layer_sizes), args.controller_hid)
                # activation embeddings
                self.act_embs = torch.nn.Embedding(len(args.shared_mlp_activations), args.controller_hid)
                act_names = args.shared_mlp_activations
                # create LSTM parameters
                self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
                # store mlp parameters
                self.network_configs['mlp'] = {'input_embs': self.input_embs,
                                               'index_attn': {'l_1': self.idx_l_1,
                                                              'l_2': self.idx_l_2,
                                                              'l_3': self.idx_l_3},
                                               'layer_embs': self.layer_embs,
                                               'act_embs': self.act_embs,
                                               'act_names': act_names,
                                               'lstm': self.lstm,
                                               'num_blocks': args.num_mlp_blocks,
                                               'layer_map': args.mlp_layer_sizes,
                                               'max_merge': args.max_mlp_merge}

            else:
                raise NotImplementedError(
                    "Networks can only be one of 'mlp' or 'rnn'. You have: {}".format(
                        network_type
                    )
                )

        # self.encoder = torch.nn.Embedding(num_total_tokens,
        #                                   args.controller_hid)
        # self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
        #
        # # TODO(brendan): Perhaps these weights in the decoder should be
        # # shared? At least for the activation functions, which all have the
        # # same size.
        # self.decoders = []
        # for idx, size in enumerate(self.num_tokens):
        #     decoder = torch.nn.Linear(args.controller_hid, size)
        #     self.decoders.append(decoder)
        #
        # self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    # TODO: Check us of this
    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        # for decoder in self.decoders:
        if 'rnn' in self.network_configs:
            for decoder in self.network_configs['rnn']['decoders']:
                decoder.bias.data.fill_(0)
        if 'mlp' in self.network_configs:
            for l_i in self.network_configs['mlp']['index_attn'].values():
                l_i.bias.data.fill_(0)



    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            # embed = self.encoder(inputs)
            embed = self.network_configs['rnn']['encoder'](inputs)
        else:
            embed = inputs

        # hx, cx = self.lstm(embed, hidden)
        hx, cx = self.network_configs['rnn']['lstm'](embed, hidden)
        logits = self.network_configs['rnn']['decoders'][block_idx](hx)
        # logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

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
        for block_idx in range(2*(self.args.num_rnn_blocks - 1) + 1):
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
                # action[:, 0] + sum(self.num_tokens[:mode]),
                action[:, 0] + sum(self.network_configs['rnn']['num_tokens'][:mode]),
                requires_grad=False)

            if mode == 0:
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)
        # Dummy layer sizes
        layer_sizes = torch.zeros(activations.shape)
        # layer_info = {'sizes':layer_sizes,
        #               'map': np.array([self.args.controller_hid])}

        dags = _construct_dags(prev_nodes,
                               activations,
                               # self.func_names,
                               self.network_configs['rnn']['func_names'],
                               self.args.num_rnn_blocks)

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))

    def sample_mlp(self, batch_size=1, with_details=False, save_dir=None):
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        index_attn = self.network_configs['mlp']['index_attn']
        l_1 = index_attn['l_1']
        l_2 = index_attn['l_2']
        l_3 = index_attn['l_3']
        lstm = self.network_configs['mlp']['lstm']

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]
        # record architecture decisions
        activations = []
        # layer_sizes = []
        entropies = []
        log_probs = []
        prev_nodes = []
        # [I, H] -> [1, I, H] -> [B, I, H]
        existing_nodes_batch = self.network_configs['mlp']['input_embs']\
            .weight\
            .view(1, self.args.num_mlp_inputs, self.args.controller_hid)\
            .expand((batch_size, self.args.num_mlp_inputs, self.args.controller_hid))

        for i in range(self.network_configs['mlp']['num_blocks']):
            # Choose input node
            # [B, H], [B, H]
            hidden_x, cell_x = lstm(inputs, hidden)
            hidden_x_batch = hidden_x.view(batch_size, 1, self.args.controller_hid)
            # update existing nodes
            # [B, I, H], [B, 1, H] --> [B, I+1, H]
            existing_nodes_batch = torch.cat(
                (existing_nodes_batch, hidden_x_batch), dim=1
            )
            # [B, I, H] + [B, 1, H] -> [B, I, H]
            ensemble_nodes = l_1(existing_nodes_batch[:, :-1, :]) + l_2(hidden_x_batch)
            # [B, I, H] x [H, 1] -> [B, I, 1]
            logits = l_3(torch.tanh(ensemble_nodes))
            # [B, I, 1] -> [B, I]
            logits = torch.reshape(logits, [batch_size, -1])
            # compute loss values
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            # sample input node from existing nodes
            input_node = probs.multinomial(num_samples=self.network_configs['mlp']['max_merge'],
                                           replacement=True).data
            # update records
            entropies.append(entropy)
            log_probs.append(log_prob)
            prev_nodes.append(input_node)
            # select input node embedding
            # [B, I+1, H] -> [B, H]
            # input_node = input_node.reshape(input_node.shape[0])
            batch_select = torch.tensor(range(batch_size)).reshape((batch_size, 1))
            input_node_emb = existing_nodes_batch[batch_select, input_node]
            input_node_emb = input_node_emb.sum(dim=1, keepdim=False)
            # input_node_emb = existing_nodes_batch[range(batch_size), input_node, :]
            """
            ##############################################################################
            #  Layer Sizes commented out for now. Some difficulty in DAG                 #
            #    Creation with averaging all unused nodes. Also this is simpler to just  #
            #    extend the model incrementally for multiple inputs                      #
            ##############################################################################
            # Choose layer size
            layer_embs = self.network_configs['mlp']['layer_embs']
            hidden_x, cell_x = lstm(input_node_emb, (hidden_x, cell_x))
            # [B, H], (L, H] -> [H, L]) -> [B, L]
            layer_logits = torch.matmul(hidden_x, layer_embs.weight.transpose(1, 0))
            # sample layer size
            layer_probs = F.softmax(layer_logits, dim=-1)
            layer_size = layer_probs.multinomial(num_samples=1).data
            # compute loss values
            layer_log_prob = F.log_softmax(layer_logits, dim=-1)
            layer_entropy = -(layer_log_prob * layer_probs).sum(1, keepdim=False)
            # update records
            entropies.append(layer_entropy)
            log_probs.append(layer_log_prob)
            layer_sizes.append(layer_size)
            # get next lstm input [B, H]
            layer_emb = layer_embs(layer_size)
            # Choose layer activation
            hidden_x, cell_x = lstm(layer_emb, (hidden_x, cell_x))
            """
            hidden_x, cell_x = lstm(input_node_emb, (hidden_x, cell_x))
            # [B, H], ([A, H] -> [H, A]) -> [B, A]
            act_embs = self.network_configs['mlp']['act_embs'].weight.transpose(1, 0)
            act_logits = torch.matmul(hidden_x, act_embs)
            act_probs = F.softmax(act_logits, dim=-1)
            act_type = act_probs.multinomial(num_samples=1).data
            # compute loss values
            act_log_prob = F.log_softmax(act_logits, dim=-1)
            act_entropy = -(act_log_prob * act_probs).sum(1, keepdim=False)
            # update records
            entropies.append(act_entropy)
            log_probs.append(act_log_prob)
            activations.append(act_type)
            # set values for LSTM loop
            act_type = act_type.reshape(act_type.shape[0])
            inputs = self.network_configs['mlp']['act_embs'](act_type)
            hidden = (hidden_x, cell_x)

        # [NumBlocks, B, NumSamples] -> [B, NumBlocks, NumSamples]
        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)
        # layer_sizes = torch.stack(layer_sizes).transpose(0, 1)

        dags = _construct_mlp_dags(prev_nodes,
                                   activations,
                                   self.network_configs['mlp']['act_names'],
                                   self.network_configs['mlp']['num_blocks'])

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_mlp_network(dag, os.path.join(save_dir, f'graph{idx}_mlp.png'))

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

if __name__ == '__main__':
    class AttributeDict(object):
        def __init__(self, d):
            d = d.copy()
            # Convert all nested dictionaries into AttrDict.
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = AttributeDict(v)
            # Convert d to AttrDict.
            self.__dict__ = d


    args = {'network_types': {'mlp', 'rnn'},
            'num_mlp_inputs': 3,
            'controller_hid': 10,
            'mlp_layer_sizes': [5],
            'shared_mlp_activations': ['tanh', 'sigmoid', 'relu', 'identity'],
            'num_mlp_blocks': 10,
            'cuda': False,
            'shared_rnn_activations': ['tanh', 'sigmoid', 'relu', 'linear'],
            'num_rnn_blocks': 10,
            'softmax_temperature': .5,
            'mode': 'train',
            'tanh_c': .5,
            'max_mlp_merge': 3}
    args = AttributeDict(args)

    save_dir = '/Users/georgestoica/Desktop/Research/ENAS-pytorch/logs/mlp'
    controller = Controller(args=args)
    # dags = controller.sample(batch_size=10, save_dir=save_dir)
    dags = controller.sample_mlp(batch_size=8, save_dir=save_dir)
