import torch.nn.functional as F
from transformers import BertForMaskedLM
from util import *
from encoders import *
import numpy as np
import time

NEG = -10000000
TOK_NC = BertTokenizer.from_pretrained("bert-base-uncased")


def logsumexp(a, b):
    return np.log(np.exp(a) + np.exp(b))


def pad(input, factor=4):
    add_size = input.size(0) % factor
    if add_size != 0:
        rem_size = factor - add_size
        return torch.cat([input, torch.zeros(rem_size, input.size(1))], dim=0)
    else:
        return input


def padding(sbatch):
    dim = sbatch[0].size(2)
    lens = [x.size(1) for x in sbatch]
    lmax = max(lens)
    padded = []
    for x in sbatch:
        pad = torch.zeros(lmax, dim)
        pad[:x.size(1),:] = x
        padded.append(pad.unsqueeze(0))
    X = torch.cat(padded, dim=0)
    return X, lens, lmax


def merge(lst):
    def merge_sub(i, lst):
        x = [i]
        while i+1 < len(lst) and lst[i+1][0] == '#':
            x.append(i+1)
            i = i+1
        return x,i+1
    x, j = merge_sub(0, lst)
    idx = [x]
    while j < len(lst):
        x, j = merge_sub(j, lst)
        idx.append(x)
    return idx


def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask


def extract(tens, mask, offset=0):
    out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[:out_lens[i]-offset])
    return torch.cat(out, dim=0)


class Attention(nn.Module):
    def __init__(self, input_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) 

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask)
        ## Add and norm
        src = Q + self.dropout(src)
        src = self.norm1(src)
        ## MLP
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        ## Add and norm
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attn


class BERTNC(nn.Module):
    def __init__(self):
        super(BERTNC, self).__init__()
        self.encoder = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).bert.embeddings
        
    def forward(self, inputs):
        return self.encoder(inputs.input_ids).permute(1,0,2), 1. - inputs.attention_mask.float()


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        model = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.encoder = model.bert
        
    def forward(self, inputs):
        output = self.encoder(**inputs)
        return output.last_hidden_state.permute(1,0,2), 1. - inputs.attention_mask

    def forward_full(self, inputs):
        output = self.encoder(**inputs)
        return output.hidden_states, 1. - inputs.attention_mask


class RNNT(nn.Module):
    def __init__(self, args):
        super(RNNT, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args['dropout'])

        if not args['unidirectional']:
            self.tNet = LstmEncoder(args['n_layer'], args['in_dim'], int(args['hid_tr']/2), dropout0=args['dropout'], dropout=args['dropout'], spec=args['deep_spec'], bidirectional=True)
        else:
            self.tNet = LstmEncoder(args['n_layer'], args['in_dim'], args['hid_tr'], dropout0=args['dropout'], dropout=args['dropout'], spec=args['deep_spec'], bidirectional=False)
        self.bottle = nn.Linear(args['hid_tr'], 768)
        self.prEmb = nn.Embedding(args['vocab_size'], 10)
        self.pNet = LstmEncoder(1, 10, args['hid_pr'], dropout0=0.01, dropout=args['dropout'], spec=args['deep_spec'], bidirectional=False)

        self.projTr = nn.Linear(768, 256)
        self.projPr = nn.Linear(args['hid_pr'], 256)
    
        self.clsProj = nn.Linear(256, args['vocab_size'])

    def forward_tr(self, x): # x is a list of sequences
        if self.args['enc_type'] != 'lstm':
            x = self.inScale(x)
        x, _ = self.tNet(x)
        x = self.bottle(self.dropout(x))
        return x

    def forward_pr(self, x): # x --> bsz, max_seq_len
        x = self.prEmb(x)
        x, _ = self.pNet(x)
        return x

    def forward_jnt(self, x, y):
        x = self.projTr(self.dropout(x)).unsqueeze(2)
        y = self.projPr(self.dropout(y)).unsqueeze(1)
        z = x + y
        return self.clsProj(torch.tanh(z))

    def forward(self, speech, tokens):
        x = self.forward_tr(speech)
        y = self.forward_pr(tokens)
        pred = self.forward_jnt(x, y)
        return pred

    def beam_search(self, speech, beam_size=1):
        self.eval()
        T = speech[0].size(0)
        D = speech[0].size(1)
        
        x = self.forward_tr(speech)
        x = self.projTr(x)

        h0 = torch.zeros(1, 1, self.args['hid_pr'])
        c0 = torch.zeros(1, 1, self.args['hid_pr'])
        beam = [((0,), 0, (h0,c0))]
        finH = {}
        cache = {}
        for i in range(4*T):
            hypL = []
            stateL = []
            for hyp, score, state in beam:
                u = len(hyp)
                t = i - u + 1
                if t == T:
                    finH[hyp] = score
                    continue
                y = cache.get(hyp)
                if y is None:
                    hypL.append(hyp)
                    stateL.append(state)

            if len(hypL) > 0:
                prev_labels = torch.LongTensor([hyp[-1:] for hyp in hypL])
                prev_state = (torch.cat([s[0] for s in stateL], dim=1), torch.cat([s[1] for s in stateL], dim=1))
                dec_time_start = time.time()
                y = self.prEmb(prev_labels) 
                y, state = self.pNet.step(y, prev_state)
                y = self.projPr(y)
                for k, hyp in enumerate(hypL):
                    cache[hyp] = y[k:k+1], (state[0][:,k:k+1,:],state[1][:,k:k+1,:])

            else_time_start = time.time()
            encL = []
            embL = []
            for hyp, score, state in beam:
                u = len(hyp)
                t = i - u + 1
                if t == T:
                    continue
                y, state = cache.get(hyp)
                embL.append(y)
                encL.append(x[:,t:t+1])
            if len(encL) > 0:
                z = torch.tanh(torch.cat(encL).unsqueeze(2) + torch.cat(embL).unsqueeze(1))
                batch_log_probs = self.clsProj(z).log_softmax(-1).view(len(encL), -1)

            new_beam = []
            k = 0
            for hyp, score, _ in beam:
                u = len(hyp)
                t = i - u + 1
                if t == T:
                    continue
                y, state = cache.get(hyp)
                log_probs = batch_log_probs[k]
                k += 1
                new_score = score + log_probs[0].item() #account for <blank> token
                new_beam.append((hyp, new_score, state))
                if t == T-1:
                    finH[hyp] = float(new_score)
                new_score = float(score)
                symL = range(1, len(ASR_ID2TOK))
                scores = log_probs[1:]
                scores += new_score
                new_hypL = [(hyp+(v,),s,state) for s,v in zip(scores.tolist(),symL)]
                new_beam += new_hypL

            if len(new_beam) == 0:
                break
            if len(new_beam) > beam_size:
                scores = [b[1] for b in new_beam]
                scores_np = np.array(scores)
                pivot = np.partition(scores_np, len(scores)-beam_size)[len(scores)-beam_size]
                beam1 = list(filter(lambda x:x[1]>=pivot, new_beam))
                uniq = {}
                for hyp, score, state in beam1:
                    if hyp in uniq:
                        uniq[hyp] = logsumexp(uniq[hyp][0], score), state
                    else:
                        uniq[hyp] = score, state
                beam = [(a, b[0], b[1]) for a, b in uniq.items()]
            else:
                beam = new_beam

        finH = sorted(finH.items(), key=lambda x: x[1], reverse=True)
        if len(finH) > 0:
            hyp, score = finH[0]
        else:
            hyp, score = [], 0

        return hyp, score