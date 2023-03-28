import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor, nn
from typing import Tuple
import torch_geometric
from utils.constants import *
from GAT_model.GAT import GAT
# class GAT(nn.Module):
#     def __init__(self, input_dim, nhid, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GraphAttentionLayer(input_dim, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)


#     def forward(self, x):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         return F.log_softmax(x, dim=1)

class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, logit_clipping=True, clip_value=10):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(enc_dim + dec_dim, dec_dim),
            nn.Tanh(),
        )
        self.v = nn.Linear(dec_dim, 1, bias=False)
        self.logit_clipping = logit_clipping
        self.clip_value = clip_value

    def forward(self, decoder_output, encoder_outputs, mask):
        # decoder_output is the output of the decoder of single step
        # encoder_outputs is a list of all the encoder outputs
        # decoder_output shape: (batch, dec_dim)
        # encoder_outputs shape: (seq_len, batch, enc_dim)

        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        decoder_output = decoder_output.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # decoder_output shape: (batch, seq_len, dec_dim)
        # encoder_outputs shape: (batch, seq_len, enc_dim)
        energy = self.attn(torch.cat((decoder_output, encoder_outputs), dim=2))
        # energy shape: (batch, seq_len, dec_dim)
        attention = self.v(energy).squeeze(2)
        if self.logit_clipping:
            attention = self.clip_value * torch.tanh(attention)
        # use mask to remove the attention weights for padded values
        attention = attention.masked_fill(mask == 0, float('-inf'))
        # attention shape: (batch, seq_len)
        return attention
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                 p):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                           dropout=p)

    def forward(self, input, mask):
        # input shape: (seq_len, batch, input_dim)
        # mask shape: (batch, seq_len)
        lengths = mask.sum(dim=1)
        # lengths shape: (batch)
        packed_inputs = pack_padded_sequence(input, lengths.to('cpu'),
                                                                enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_inputs)
        output, _ = pad_packed_sequence(packed_outputs)
        # output shape: (seq_len, batch, hidden_dim)
        # hidden shape: (n_layers, batch, hidden_dim) cell shape: (n_layers, batch, hidden_dim)
        # if encoder is bidirectional
        # pass hidden and cell through a linear layer to match decoder hidden and cell dimension
        seq_len = input.shape[0]
        # reshape output to (seq_len, batch, hidden_dim)
        return output, (hidden, cell)


class Decoder(nn.Module):
    # module for single decoder stage
    def __init__(self, embb_dim, hidd_dim, n_layers, p):
        super(Decoder, self).__init__()
        self.output_dim = embb_dim
        self.hidden_dim = hidd_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embb_dim, hidd_dim, n_layers, dropout=p)

    def forward(self, input, hidden, cell):
        # input shape: (sequence length, batch, input_dim)
        # hidden shape: (n_layers, batch, hidden_dim)
        # encoder_outputs shape: (seq_len, batch, hidden_dim)
        # output shape: (batch, input_dim)
        input = input.unsqueeze(0)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = output.squeeze(0)
        return output, (hidden, cell)

def rearrange(x:torch.Tensor, beams_buf:torch.Tensor)->torch.Tensor:
    batch_size, num_samples = beams_buf.size()
    if x.dim() == 1:
        return x.view(batch_size, num_samples).gather(1,beams_buf).view(-1)
    elif x.dim() == 2:
        return x.view(batch_size, num_samples, -1)\
            .gather(1,beams_buf.unsqueeze(-1).expand(-1, -1, x.size(-1)))\
                .view(-1, x.size(-1))
    elif x.dim() == 3:
        return x.permute(1,0,2)\
            .view(batch_size, num_samples, x.size(0), x.size(2))\
            .gather(1,beams_buf.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(0), x.size(2)))\
                .view(-1, x.size(0), x.size(2))\
                    .permute(1,0,2)\
                        .contiguous()

class PointerNet(nn.Module):
    def __init__(self,num_GAT_layers, num_heads_per_layer,num_features_per_GATlayer, GAT_dropout,
                 ENC,input_dim, embb_dim, hidd_dim, dec_layers, dropout, mh_attn,device, logit_clipping=True, decoding_type = 'sampling'):
        print("using lstm")
        super().__init__()
        self.gat: GAT = GAT(num_of_layers= num_GAT_layers,
            num_heads_per_layer= num_heads_per_layer,
            num_features_per_layer=num_features_per_GATlayer,
            add_skip_connection=False,
            bias=True,
            dropout=GAT_dropout,
            layer_type=LayerType.IMP3, # GAT IMP3 is used here
            log_attention_weights=False)  # no need to store attentions, used only in playground.py for visualizations
        if (ENC == True):
            self.encoder: Encoder = Encoder(embb_dim, hidd_dim, dec_layers, dropout)
        self.decoder: Decoder = Decoder(embb_dim, hidd_dim, dec_layers, dropout)
        self.attn: Attention = Attention(hidd_dim, hidd_dim, logit_clipping)
        self.initial_decoder_input = nn.Parameter(torch.zeros(1, embb_dim))
        self.device = device
        self.decoding_type = decoding_type
        self.embb_dim = embb_dim
        self.mh_attn = mh_attn
        self.ENC = ENC
    def preprocess(self, input, mask) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=1, dtype=torch.long)
        # seq_len is the length of the longest sequence in the batch
        seq_len = torch.max(lengths).item()
        # reshape input and mask to remove extra padding
        # first dimension of the input is seq_len
        input = input[:seq_len]
        # second dimension of the mask is seq_len
        mask = mask[:, :seq_len]
        return input, mask


    def forward(self, input: Tensor, num_samples=1):
        batch_size = 1
        seq_len = input.x.shape[0]
        # input.x = torch.ones(seq_len,seq_len,device=self.device) # for testing 
        GAT_outputs,_ = self.gat([input.x,input.edge_index]) # encoder_outputs shape: (num_nodes, GAT_out_dim)
        batched_embeddings, mask = torch_geometric.utils.to_dense_batch(GAT_outputs, input.batch)
        batched_embeddings = batched_embeddings.squeeze()
        batched_embeddings = batched_embeddings.unsqueeze(-2)
        # batched_embeddings size: batch_size, num_nodes, out_dim(GAT)
        # if len(input.x.shape) == 2:
        #     encoder_outputs=torch.unsqueeze(encoder_outputs,0).repeat(1,  1, 1)
        # sample n_samples mapping solutions for each sequence in the batch
        # input, mask = self.preprocess(input, mask)
        # Tensor to store the predicted mapping
        # predicted_mappings shape: (batch * num_samples, seq_len)
        predicted_mappings = torch.zeros(batch_size * num_samples, seq_len, dtype=torch.int64).to(self.device)
        if(self.ENC == True):
            encoder_outputs, (hidden, cell) = self.encoder(batched_embeddings, mask)
        else:
            encoder_outputs = batched_embeddings
            hidden = torch.zeros(batch_size,1,encoder_outputs.shape[2]).to(self.device)
            cell = torch.zeros(batch_size,1,encoder_outputs.shape[2]).to(self.device)
        # first input should be a part of model learnable parameters
        decoder_input = self.initial_decoder_input.repeat(batch_size,1)
        # hidden = torch.zeros(batch_size,1,encoder_outputs.shape[1]).to(self.device)
        # cell = torch.zeros(batch_size,1,encoder_outputs.shape[1]).to(self.device)
        # mask to be used while calculating attention weights
        mask_decoding = mask.clone()
        log_probs_sum = torch.zeros(batch_size * num_samples, dtype=torch.float32).to(self.device)
        # log_probs_sum_tsv = torch.zeros(batch_size * num_samples, dtype=torch.float32).to(self.device)
        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            logits = self.attn(output, encoder_outputs, mask_decoding)
            # logits shape: (batch * num_samples, seq_len)
            log_probs = F.log_softmax(logits, dim=1)
            # log_probs shape: (batch * num_samples, seq_len)
            if t == 0:
                # log_probs shape: (batch , seq_len)
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                    # selected_indices shape: (batch * num_samples,)
                        scores = log_probs + gumbel_like(log_probs)
                        # scores shape: (batch * num_samples, seq_len)
                    elif self.decoding_type == 'greedy':
                        scores = log_probs
                    _, selected_indices = torch.topk(scores, min(num_samples, scores.size(-1)), dim=-1)
                    # selected_indices shape: (batch, min(num_samples, seq_len))
                    if num_samples > log_probs.size(1):
                        # pad second dimension with -1 so that the shape becomes (batch, num_samples)
                        selected_indices = F.pad(selected_indices, (0, num_samples - scores.size(1)), 'constant', -1)
                    selected_indices = selected_indices.view(-1)
                else:
                    selected_indices = torch.multinomial(log_probs.exp(), num_samples, replacement=True).long().view(-1)
                log_probs = log_probs.repeat_interleave(num_samples, dim=0)
                # make log_probs -inf for the values which are -1 in selected_indices
                log_probs = log_probs.masked_fill((selected_indices == -1)\
                    .unsqueeze(-1), float('-inf'))
                if self.decoding_type == 'sampling-w/o-replacement':
                    scores = scores.repeat_interleave(num_samples, dim=0)
                    scores = scores.masked_fill((selected_indices == -1)\
                        .unsqueeze(-1), float('-inf'))
                mask_decoding = mask_decoding.repeat_interleave(num_samples, dim=0)
                # mask_decoding shape: (batch * num_samples, seq_len)
                hidden = hidden.repeat_interleave(num_samples, dim=1)
                cell = cell.repeat_interleave(num_samples, dim=1)
                encoder_outputs = encoder_outputs.repeat_interleave(num_samples, dim=1)
            else:
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores, _ = gumbel_with_maximum(log_probs + log_probs_sum.unsqueeze(-1), g_log_probs)
                        
                    elif self.decoding_type == 'greedy':
                        scores = log_probs + log_probs_sum.unsqueeze(-1) 
                    scores_per_batch = scores.view(batch_size, -1)
                    _, indices_buf = torch.topk(scores_per_batch, num_samples, dim=1)
                    # indices_buf shape: (batch, min(num_samples, scores.size(1)))
                    beams_buf = torch.div(indices_buf, seq_len,rounding_mode='floor')
                    indices_buf = indices_buf.fmod(seq_len)
                    selected_indices = indices_buf.view(-1)
                    predicted_mappings[:,:t] = rearrange(predicted_mappings[:,:t], beams_buf)
                    log_probs_sum = rearrange(log_probs_sum, beams_buf)
                    mask_decoding = rearrange(mask_decoding, beams_buf)
                    log_probs = rearrange(log_probs, beams_buf)
                    hidden = rearrange(hidden, beams_buf)
                    cell = rearrange(cell, beams_buf)
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores = rearrange(scores, beams_buf)
                else:
                    selected_indices = torch.multinomial(log_probs.exp(), 1).long().squeeze(1)
            predicted_mappings[:, t] = selected_indices
            gather_indices = selected_indices.unsqueeze(-1).clone()
            gather_indices[gather_indices == -1] = 0 
            curr_log_probs = log_probs.gather(1, gather_indices).squeeze(-1) \
                * mask.repeat_interleave(num_samples, dim=0)[:, t]
            log_probs_sum += curr_log_probs
            if self.decoding_type == 'sampling-w/o-replacement':
                # gumbel perturbed log probabilities of partial sequences
                g_log_probs = scores.gather(1, gather_indices).squeeze(-1) \
                    * mask.repeat_interleave(num_samples, dim=0)[:, t]
            # decoder_input shape: (batch, input_dim)

            decoder_input = batched_embeddings.repeat_interleave(num_samples, dim=1)[gather_indices.squeeze(-1), torch.arange(batch_size * num_samples)]
        
            # decoder_input = torch.gather(input=batched_embeddings,dim=2,index=predicted_mappings[:,t].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.embb_dim))
            # decoder_input = decoder_input.squeeze()
            # update the mask_decoding to remove the pointed inputs
            mask_decoding.scatter_(1, gather_indices, 0)
        
        # assign -1 to the mappings corresponding to the padded values to denote that it is invalid
        predicted_mappings = predicted_mappings.masked_fill(mask.repeat_interleave(num_samples, dim=0) == 0, -1)
        # mask_tsv = mask[0][0:tsv_num].unsqueeze(0)
        # predicted_tsvs = predicted_tsvs.masked_fill(mask_tsv.repeat_interleave(num_samples, dim=0) == 0, -1)
        # # reshape the predicted_mappings to (batch_size, seq_len, n_samples)
        # predicted_mappings = predicted_mappings.view(batch_size, num_samples, seq_len)
        predicted_mappings = predicted_mappings.view(batch_size, num_samples, seq_len).transpose(0, 1).reshape(-1,seq_len)
        # predicted_tsvs = predicted_tsvs.view(batch_size, num_samples, tsv_num).transpose(0, 1).reshape(-1,tsv_num)
        log_probs_sum = log_probs_sum.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
        # log_probs_sum_tsv = log_probs_sum_tsv.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
        if self.decoding_type == 'sampling-w/o-replacement':
            g_log_probs = g_log_probs.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
            # g_log_probs_tsv = g_log_probs_tsv.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
            # return predicted_mappings, log_probs_sum, g_log_probs, predicted_tsvs, log_probs_sum_tsv,g_log_probs_tsv
            return predicted_mappings, log_probs_sum, g_log_probs
        else:
            return predicted_mappings, log_probs_sum
