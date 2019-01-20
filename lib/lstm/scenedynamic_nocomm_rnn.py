import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from typing import Optional, Tuple

from lib.sparse_targets import FrequencyBias
from lib.fpn.box_utils import nms_overlaps
from lib.word_vectors import obj_edge_vectors
from .highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
import numpy as np

def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.autograd.Variable):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask

class DecoderRNN(torch.nn.Module):
    def __init__(self, classes, rel_classes, embed_dim, obj_dim, inputs_dim, 
                 hidden_dim, pooling_dim, recurrent_dropout_probability=0.2,
                 use_highway=True, use_input_projection_bias=True, use_vision=True, 
                 use_bias=True, use_tanh=True, limit_vision=True, sl_pretrain=False, num_iter=-1):
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        """
        super(DecoderRNN, self).__init__()

        self.rel_embedding_dim = 100
        self.classes = classes
        self.rel_classes = rel_classes
        embed_vecs = obj_edge_vectors(['start'] + self.classes, wv_dim=100)
        self.obj_embed = nn.Embedding(len(self.classes), embed_dim)
        self.obj_embed.weight.data = embed_vecs

        embed_rels = obj_edge_vectors(self.rel_classes, wv_dim=self.rel_embedding_dim)
        self.rel_embed = nn.Embedding(len(self.rel_classes), self.rel_embedding_dim) 
        self.rel_embed.weight.data = embed_rels

        self.embed_dim = embed_dim
        self.obj_dim = obj_dim
        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.pooling_dim = pooling_dim
        self.nms_thresh = 0.3

        self.use_vision = use_vision
        self.use_bias = use_bias
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.sl_pretrain = sl_pretrain
        self.num_iter = num_iter

        self.recurrent_dropout_probability=recurrent_dropout_probability
        self.use_highway=use_highway
        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        if use_highway:
            self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size,
                                                   bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size,
                                                   bias=True)
        else:
            self.input_linearity = torch.nn.Linear(self.input_size, 4 * self.hidden_size,
                                                   bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(self.hidden_size, 4 * self.hidden_size,
                                                   bias=True)

        # self.obj_in_lin = torch.nn.Linear(self.rel_embedding_dim, self.rel_embedding_dim, bias=True)

        self.out = nn.Linear(self.hidden_size, len(self.classes))
        self.reset_parameters()

        # For relation predication
        embed_vecs2 = obj_edge_vectors(self.classes, wv_dim=embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_classes, embed_dim)
        self.obj_embed2.weight.data = embed_vecs2.clone()

        # self.post_lstm = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
        self.post_lstm = nn.Linear(self.obj_dim+2*self.embed_dim+128, self.pooling_dim * 2)
        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.
        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_size)) ######## there may need more consideration
        self.post_lstm.bias.data.zero_()

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()

            # simple relation model
            from dataloaders.visual_genome import VG
            from lib.get_dataset_counts import get_counts, box_filter
            fg_matrix, bg_matrix = get_counts(train_data=VG.splits(num_val_im=5000, 
                                                                   filter_non_overlap=True,
                                                                   filter_duplicate_rels=True,
                                                                   use_proposals=False)[0], must_overlap=True)
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix

            # TRYING SOMETHING NEW.
            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())

            prob_matrix[:,:,0] = 0 # Zero out BG
            self.prob_matrix = prob_matrix


    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    @property
    def input_size(self):
        return self.inputs_dim + self.obj_embed.weight.size(1)

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        """
        Does the hairy LSTM math
        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
            highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def get_rel_dist(self, obj_preds, obj_feats, rel_inds, vr=None):
        obj_embed2 = self.obj_embed2(obj_preds)
        edge_ctx = torch.cat((obj_embed2, obj_feats), 1)

        edge_rep = self.post_lstm(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.pooling_dim)

        subj_rep = edge_rep[:, 0]
        obj_rep = edge_rep[:, 1]

        prod_rep = subj_rep[rel_inds[:, 1]] * obj_rep[rel_inds[:, 2]]

        if self.use_vision:
            if self.limit_vision:
                # exact value TBD
                prod_rep = torch.cat((prod_rep[:,:2048] * vr[:,:2048], prod_rep[:,2048:]), 1)
            else:
                prod_rep = prod_rep * vr

        if self.use_tanh:
            prod_rep = F.tanh(prod_rep)

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(torch.stack((
                obj_preds[rel_inds[:, 1]],
                obj_preds[rel_inds[:, 2]],
            ), 1))

        return rel_dists

    def get_freq_rel_dist(self, obj_preds, rel_inds):
        """
        Baseline: relation model
        """
        rel_dists = self.freq_bias.index_with_labels(torch.stack((
            obj_preds[rel_inds[:, 1]],
            obj_preds[rel_inds[:, 2]],
        ), 1))

        return rel_dists

    def get_simple_rel_dist(self, obj_preds, rel_inds):

        obj_preds_np = obj_preds.cpu().numpy()
        rel_inds_np = rel_inds.cpu().numpy()

        rel_dists_list = []
        o1o2 = obj_preds_np[rel_inds_np][:, 1:]
        for o1, o2 in o1o2:
            rel_dists_list.append(self.prob_matrix[o1, o2])

        assert len(rel_dists_list) == len(rel_inds)
        return Variable(torch.from_numpy(np.array(rel_dists_list)).cuda(obj_preds.get_device())) # there is no gradient for this type of code

    def forward(self,  # pylint: disable=arguments-differ
                # inputs: PackedSequence,
                sequence_tensor,
                rel_inds,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                labels=None, boxes_for_nms=None, vr=None):

        # get the relations for each object
        # numer = torch.arange(0, rel_inds.size(0)).long().cuda(rel_inds.get_device())

        # objs_to_outrels = sequence_tensor.data.new(sequence_tensor.size(0), 
        #                                             rel_inds.size(0)).zero_()
        # objs_to_outrels.view(-1)[rel_inds[:, 1] * rel_inds.size(0) + numer] = 1
        # objs_to_outrels = Variable(objs_to_outrels)

        # objs_to_inrels = sequence_tensor.data.new(sequence_tensor.size(0), rel_inds.size(0)).zero_()
        # objs_to_inrels.view(-1)[rel_inds[:, 2] * rel_inds.size(0) + numer] = 1
        # # average the relations for each object, and add "non relation" to the one with on relation communication
        # # test8 / test10 need comments
        # objs_to_inrels = objs_to_inrels / (objs_to_inrels.sum(1) + 1e-8)[:, None]
        # objs_to_inrels = Variable(objs_to_inrels)

        batch_size = sequence_tensor.size(0)

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = Variable(sequence_tensor.data.new()
                                                  .resize_(batch_size, self.hidden_size).fill_(0))
            previous_state = Variable(sequence_tensor.data.new()
                                                 .resize_(batch_size, self.hidden_size).fill_(0))
        else:
            assert len(initial_state) == 2
            previous_state = initial_state[0].squeeze(0)
            previous_memory = initial_state[1].squeeze(0)

        # 'start'
        previous_embed = self.obj_embed.weight[0, None].expand(batch_size, 100)

        # previous_comm_info = Variable(sequence_tensor.data.new()
        #                                             .resize_(batch_size, 100).fill_(0))

        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, previous_memory)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists_list = []
        out_commitments_list = []

        end_ind = 0
        for i in range(self.num_iter):

            # timestep_input = torch.cat((sequence_tensor, previous_embed, previous_comm_info), 1)
            timestep_input = torch.cat((sequence_tensor, previous_embed), 1)

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)

            pred_dist = self.out(previous_state)
            out_dists_list.append(pred_dist)

            # if self.training:
            #     labels_to_embed = labels.clone()
            #     # Whenever labels are 0 set input to be our max prediction
            #     nonzero_pred = pred_dist[:, 1:].max(1)[1] + 1
            #     is_bg = (labels_to_embed.data == 0).nonzero()
            #     if is_bg.dim() > 0:
            #         labels_to_embed[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
            #     out_commitments_list.append(labels_to_embed)
            #     previous_embed = self.obj_embed(labels_to_embed+1)
            # else:
            #     out_dist_sample = F.softmax(pred_dist, dim=1)
            #     # if boxes_for_nms is not None:
            #     #     out_dist_sample[domains_allowed[i] == 0] = 0.0

            #     # Greedily take the max here amongst non-bgs
            #     best_ind = out_dist_sample[:, 1:].max(1)[1] + 1

            #     # if boxes_for_nms is not None and i < boxes_for_nms.size(0):
            #     #     best_int = int(best_ind.data[0])
            #     #     domains_allowed[i:, best_int] *= (1 - is_overlap[i, i:, best_int])
            #     out_commitments_list.append(best_ind)
            #     previous_embed = self.obj_embed(best_ind+1)
            if self.training and (not self.sl_pretrain):
                import pdb; pdb.set_trace()
                out_dist_sample = F.softmax(pred_dist, dim=1)
                sample_ind = out_dist_sample[:, 1:].multinomial(1)[:, 0] + 1 # sampling at training stage
                out_commitments_list.append(sample_ind)
                previous_embed = self.obj_embed(sample_ind+1)
            else:
                out_dist_sample = F.softmax(pred_dist, dim=1)
                # best_ind = out_dist_sample[:, 1:].max(1)[1] + 1
                # debug
                best_ind = out_dist_sample.max(1)[1]  ###########################
                out_commitments_list.append(best_ind)
                previous_embed = self.obj_embed(best_ind+1)

            # calculate communicate information
            # rel_dists = self.get_rel_dist(best_ind, sequence_tensor, rel_inds, vr)
            # all_comm_info = rel_dists @ self.rel_embed.weight

            # obj_rel_weights = sequence_tensor @ torch.transpose(self.obj_rel_att.weight, 1, 0) @ torch.transpose(all_comm_info, 1, 0)
            # masked_objs_to_inrels = obj_rel_weights * objs_to_inrels
            # objs_to_inrels = masked_objs_to_inrels / (masked_objs_to_inrels.sum(1) + 1e-8)[:, None]

            # previous_comm_info = self.obj_in_lin(objs_to_inrels @ all_comm_info)

        out_dists = out_dists_list[-1]
        out_commitments = out_commitments_list[-1]
        # Do NMS here as a post-processing step
        """
        if boxes_for_nms is not None and not self.training:
            is_overlap = nms_overlaps(boxes_for_nms.data).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh
            # is_overlap[np.arange(boxes_for_nms.size(0)), np.arange(boxes_for_nms.size(0))] = False
            out_dists_sampled = F.softmax(out_dists).data.cpu().numpy()
            out_dists_sampled[:,0] = -1.0 # change 0.0 to 1.0 for the bug when the score for bg is almost 1.
            out_commitments = out_commitments.data.new(len(out_commitments)).fill_(0)
            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = -1.0 #0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample
            out_commitments = Variable(out_commitments)
        """
        # rel_dists = self.get_rel_dist(out_commitments, sequence_tensor, rel_inds, vr)
        # simple model
        # import pdb; pdb.set_trace()
        # rel_dists = self.get_freq_rel_dist(out_commitments, rel_inds)
        
        rel_dists = self.get_simple_rel_dist(out_commitments.data, rel_inds)

        return out_dists_list, out_commitments_list, None, \
                    out_dists, out_commitments, rel_dists