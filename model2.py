from torch.nn.utils.rnn import pack_sequence
from speechbrain.nnet.RNN import AttentionalRNNDecoder
from lstm_util import *
import pdb
import numpy as np
import torch
import torch.nn as nn
import random
import copy
import torch.nn.functional as F


def mixup(r_t, r_s, targets_t, targets_s, device, beta=2.0):
    M = r_t.size(0)
    N = r_s.size(0)
    S = r_t.size(1)
    C = targets_t.size(1)
    idx = list(range(N))
    random.shuffle(idx)
    idx = idx[:100]
    r_t = r_t.unsqueeze(1)
    r_s = r_s.unsqueeze(0)
    tgt_t = targets_t.unsqueeze(1)
    tgt_s = targets_s.unsqueeze(0)
    temp = torch.zeros(M,N,1)
    temp[:] = beta
    beta_list = temp.tolist()
    lam = np.random.beta(beta_list, beta_list)
    lam = torch.from_numpy(np.maximum(lam, 1. - lam)).float().to(device)
    pdb.set_trace()
    r_mix = (lam) * r_t + (1-lam)*r_s
    tgt_mix = (lam) * tgt_t + (1-lam)*tgt_s
    return r_mix.view(-1, S), tgt_mix.view(-1, C)


def mixup2(rep, labels, device):
    targets = mhot_tgt(labels.tolist(), 7).to(device)
    high, low = 10, 0.1
    beta = []
    for i in labels:
        if i == 0:
            beta.append(low)
        else:
            beta.append(high)
    idx = list(range(rep.size(0)))
    random.shuffle(idx)
    rep_ = rep[idx]
    targets_ = targets[idx]
    lam = np.random.beta(beta, beta)
    lam = torch.from_numpy(np.maximum(lam, 1. - lam)).float().to(device)
    lam = lam.unsqueeze(1)
    rep_mix = lam * rep + (1.-lam) * rep_
    targets_mix = lam * targets + (1.-lam)*targets_
    return rep_mix, targets_mix


def mhot_tgt(tlist, ncls):
    tgt = torch.zeros(len(tlist), ncls)
    for i,t in enumerate(tlist):
        tgt[i][t] = 1.
    return tgt


def unpack_speech(input_s, layer=3):
    for i in range(layer):
        timestep = input_s.size(0)
        feature_dim = input_s.size(1)
        input_s = input_s.contiguous().view(int(timestep/2), feature_dim*2)
    return input_s


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def mean_pool(tens, mask):
    return (tens*(1.-mask).t().unsqueeze(2)).sum(dim=0) / (1.-mask).sum(dim=1,keepdim=True)


def extract(tens, mask):
    out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[:out_lens[i]])
    return torch.cat(out, dim=0)


def merge(tens, merge_idx):
    out = []
    for bat, idx_bat in enumerate(merge_idx):
        for seq, idx_seq in enumerate(idx_bat):
            out.append(tens[bat][idx_seq].mean(dim=0, keepdim=True))
    return torch.cat(out, dim=0)


def merge_keep_seq(tens, merge_idx):
    out = []
    for bat, idx_bat in enumerate(merge_idx):
        seq = []
        for _, idx_seq in enumerate(idx_bat):
            seq.append(tens[bat][idx_seq].mean(dim=0, keepdim=True))
        out.append(torch.cat(seq, dim=0))
    out = pack_sequence(out, enforce_sorted=False)
    out, lens = pad_packed_sequence(out, batch_first=True)
    return out, get_mask(lens)


def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask


class Listener(nn.Module):
    def __init__(self, input_dim, pyr_layer, nlayer, dropout=0.1):
        super(Listener, self).__init__()
        self.pyr_layer = pyr_layer
        self.p_encoder = pLSTM(input_dim, pyr_layer, dropout=dropout)
        self.encoder = CustomLSTM((2**pyr_layer)*input_dim, nlayer, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens):
        lens_org = (copy.deepcopy(lens) / (2**self.pyr_layer)).long()
        out_pyr, lens = self.p_encoder(input_x, lens)
        out_lstm, _ = self.encoder(out_pyr, lens)
        return out_lstm, lens_org


class Attention(nn.Module):
    def __init__(self, d_model, nhead=1, dim_feedforward=1280, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn


class ConEncoder(nn.Module):
    def __init__(self, input_dim=20, output_dim=128, dropout=0.5):
        super(ConEncoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, output_dim, 1, bidirectional=True)
        self.tdd = nn.Linear(2*output_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.lin = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, lens):
        lens_org = (copy.deepcopy(lens)).long()
        lens = lens.cpu()
        lens = lens + input.size(1) - lens.max()
        # pack sequence
        pack = pack_padded_sequence(input, lens, batch_first=True, enforce_sorted=False)
        # forward pass - LSTM
        self.encoder.flatten_parameters()
        output, hidden = self.encoder(pack)
        # pad packed seq output of LSTM
        out_pad, lens = pad_packed_sequence(output, batch_first=True)
        # Time distributed dense layer
        out_tdnn = F.relu(self.tdd(out_pad))
        out_bn1 = self.dropout(self.bn1(out_tdnn.permute(0,2,1)).permute(0,2,1))
        out_bn2 = self.dropout(self.bn2(F.relu(self.lin(out_bn1)).permute(0,2,1)).permute(0,2,1))
        return out_bn2, lens_org


class DisEncoder(nn.Module):
    def __init__(self, embed_dim=128, dropout=0.5):
        super(DisEncoder, self).__init__()
        self.embedding = nn.Embedding(32, embed_dim)
        self.encoder = nn.LSTM(embed_dim, embed_dim, 1, bidirectional=True)
        self.tdd = nn.Linear(2*embed_dim, embed_dim)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens):
        lens_org = (copy.deepcopy(lens)).long()
        lens = lens.cpu()
        lens = lens + input_x.size(1) - lens.max()
        embed_in = self.embedding(input_x)
        # pack sequence
        pack = pack_padded_sequence(embed_in, lens, batch_first=True, enforce_sorted=False)
        # forward pass - LSTM
        self.encoder.flatten_parameters()
        output, hidden = self.encoder(pack)
        # pad packed seq output of LSTM
        out_pad, lens = pad_packed_sequence(output, batch_first=True)
        # Time distributed dense layer
        out_tdnn = F.relu(self.tdd(out_pad))
        out_bn1 = self.dropout(self.bn1(out_tdnn.permute(0,2,1)).permute(0,2,1))
        out_bn2 = self.dropout(self.bn2(F.relu(self.lin(out_bn1)).permute(0,2,1)).permute(0,2,1))
        return out_bn2, lens_org


class PhonetCLS(nn.Module):
    def __init__(self, config):
        super(PhonetCLS, self).__init__()
        ncls = config['nclasses']
        nFeat = 128
        self.mfcc_enc = ConEncoder(input_dim=20, output_dim=nFeat)
        self.post_enc = ConEncoder(input_dim=18, output_dim=nFeat)
        self.phon_enc = DisEncoder(embed_dim=nFeat)
        if config['multi-gpu']:
            self.mfcc_enc = nn.DataParallel(self.mfcc_enc, device_ids=[0,1])
            self.post_enc = nn.DataParallel(self.post_enc, device_ids=[0,1])
            self.phon_enc = nn.DataParallel(self.phon_enc, device_ids=[0,1])
        self.lin1 = nn.Linear(3*nFeat, 3*nFeat)
        self.bn1 = nn.BatchNorm1d(3*nFeat)
        self.lin2 = nn.Linear(3*nFeat, 3*nFeat)
        self.bn2 = nn.BatchNorm1d(3*nFeat)
        self.cls = nn.Linear(3*nFeat, ncls)
        self.dropout = nn.Dropout(0.5)

    def forward(self, mfcc, post, phon, mfcc_len, post_len, phon_len):
        mfcc_out, len_mfcc = self.mfcc_enc(mfcc, mfcc_len)
        post_out, len_post = self.post_enc(post, post_len)
        phon_out, len_phon = self.phon_enc(phon, phon_len)
        mask_mfcc, mask_post, mask_phon = get_mask(len_mfcc.cpu().tolist()).to(mfcc_out.get_device()), get_mask(len_post.cpu().tolist()).to(mfcc_out.get_device()), get_mask(len_phon.cpu().tolist()).to(mfcc_out.get_device())

        mask_mfcc = (-100000 * mask_mfcc).unsqueeze(-1)
        mask_post = (-100000 * mask_post).unsqueeze(-1)
        mask_phon = (-100000 * mask_phon).unsqueeze(-1)

        feat_mfcc = (mfcc_out + mask_mfcc).max(dim=1).values
        feat_post = (post_out + mask_post).max(dim=1).values
        feat_phon = (phon_out + mask_phon).max(dim=1).values

        feat = torch.cat([feat_mfcc, feat_post, feat_phon], dim=1)
        out_1 = F.relu(self.lin1(feat))
        out_2 = self.dropout(self.bn1(out_1))
        out_3 = self.dropout(self.bn2(F.relu(self.lin2(out_2))))

        return self.cls(out_3)


class Reader(nn.Module):
    def __init__(self, embed_dim, vocab_size, dropout=0.1):
        super(Reader, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, 2*embed_dim, 1, bidirectional=False)
        self.norm = nn.LayerNorm(2*embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens): # bsz, seq_len
        lens_org = (copy.deepcopy(lens)).long()
        #pdb.set_trace()
        lens = lens.cpu()
        lens = lens + input_x.size(1) - lens.max()
        embed_in = self.embedding(input_x)
        # pack sequence
        pack = pack_padded_sequence(embed_in, lens, batch_first=True, enforce_sorted=False)
        # forward pass - LSTM
        self.encoder.flatten_parameters()
        output, hidden = self.encoder(pack)
        # pad packed seq output of LSTM
        out_pad, lens = pad_packed_sequence(output, batch_first=True)
        output = self.norm(out_pad)
        return output, lens_org


class ASR(nn.Module):
    def __init__(self, config):
        super(ASR, self).__init__()
        self.reader = Reader(config['embed_dim'], config['vocab_size'], config['dropout'])
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['nlayer'], config['dropout'])

        if config['multi-gpu']:
            self.reader = nn.DataParallel(self.reader, device_ids=[0,1])
            self.listener = nn.DataParallel(self.listener, device_ids=[0,1])

        attention_indim = (2**config['pyr_layer'])*config['input_dim']
        self.attention = Attention(attention_indim, nhead=config['nhead'], dim_feedforward=2*attention_indim, dropout=config['dropout'])

        self.rnn = nn.LSTM(2*attention_indim, attention_indim, 1, bidirectional=False)
        self.classifier_tok = nn.Linear(attention_indim, config['vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, input_s, input_t, lens_s, lens_t, merge_idx=None):
        listened, lens_s_ = self.listener(input_s, lens_s)
        read, lens_t_ = self.reader(input_t, lens_t)
        mask_s, mask_t = get_mask(lens_s_.cpu().tolist()).to(read.get_device()), get_mask(lens_t_.cpu().tolist()).to(read.get_device())

        aligned_tq, attn_tq = self.attention(read.permute(1,0,2), listened.permute(1,0,2), mask_s.bool())

        self.rnn.flatten_parameters()
        out_feat, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read.permute(1,0,2)], dim=2)))
        out = extract(self.classifier_tok(self.dropout(out_feat)).permute(1,0,2), mask_t.long())

        return out, None, attn_tq


class Classifier(nn.Module):
    def __init__(self, indim, outdim):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(indim, indim)
        self.l2 = nn.Linear(indim, indim)
        self.l3 = nn.Linear(indim, outdim)
        self.dropout = nn.Dropout(0.1)

    def mfw(self, X, label, gamma=12350.35015119583, mu=429.7652650998209, beta=0.01):
        mapping = {0:35688, 1:325, 2:1118, 3:567, 4:271, 5:151, 6:9}
        lam = torch.from_numpy(np.random.beta(2.0, 2.0, (X.size(0), 1))).float()
        sn = torch.tensor([mapping[x] for x in label.tolist()]).unsqueeze(1).float()
        sn = 0.5*torch.sigmoid((sn-mu)/(beta*gamma))
        lam = sn*lam
        lam = lam.to(X.get_device())
        idx = list(range(X.size(0)))
        random.shuffle(idx)
        X = (1. - lam)*X + lam*X[idx]
        return X

    def forward(self, input, label=None):
        mix_layer = random.sample([0, 1, 2], 1)
        if mix_layer == 0 and label is not None:
            input = self.mfw(input, label)
        op1 = torch.tanh(self.l1(self.dropout(input)))
        if mix_layer == 1 and label is not None:
            op1 = self.mfw(op1, label)
        op2 = torch.tanh(self.l2(self.dropout(op1)))
        if mix_layer == 2 and label is not None:
            op2 = self.mfw(op2, label)
        op3 = self.l3(self.dropout(op2))
        return op3


class WordCLS(nn.Module):
    def __init__(self, config):
        super(WordCLS, self).__init__()
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['nlayer'], config['dropout'])
        if config['multi-gpu']:
            self.listener = nn.DataParallel(self.listener, device_ids=[0,1])

        indim = (2**config['pyr_layer'])*config['input_dim']
        self.classifier = nn.Linear(indim, config['nclasses'])
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, input_s, lens_s, label=None):
        listened, lens_s_ = self.listener(input_s, lens_s)
        mask_s = get_mask(lens_s_.cpu().tolist()).to(listened.get_device())
        #mask_s = 1. - mask_s
        #feat = (listened * mask_s.unsqueeze(-1)).sum(dim=1) / mask_s.sum(dim=1, keepdim=True)
        mask_s = (-100000 * mask_s).unsqueeze(-1)
        feat = (listened + mask_s).max(dim=1).values
        return self.classifier(self.dropout(feat))


class Detector(nn.Module):
    def __init__(self, config):
        super(Detector, self).__init__()
        self.reader = Reader(config['embed_dim'], config['vocab_size'], config['dropout'])
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['nlayer'], config['dropout'])

        if config['multi-gpu']:
            self.reader = nn.DataParallel(self.reader, device_ids=[0,1])
            self.listener = nn.DataParallel(self.listener, device_ids=[0,1])

        attention_indim = (2**config['pyr_layer'])*config['input_dim']
        self.attention = Attention(attention_indim, nhead=config['nhead'], dim_feedforward=2*attention_indim, dropout=config['dropout'])

        self.rnn = nn.LSTM(2*attention_indim, attention_indim, 1, bidirectional=False)
        self.rnn_cls = nn.LSTM(attention_indim, attention_indim, 1, bidirectional=False)
        self.rnn_cls_2 = nn.LSTM(attention_indim, attention_indim, 1, bidirectional=False)

        self.classifier_tok = nn.Linear(attention_indim, config['vocab_size'])
        self.classifier_cls = nn.Linear(attention_indim, config['nclasses'])
        self.classifier_det = nn.Linear(attention_indim, 2)
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def mfw(self, X, label, mapping=None, gamma=12350.35015119583, mu=429.7652650998209, beta=0.01):
        if mapping is None:
            mapping = {0:35688, 1:325, 2:1118, 3:567, 4:271, 5:151, 6:9}#{0:36312, 1:481, 3:337, 4:395, 5:3341}#{0:35688, 1:325, 2:1118, 3:567, 4:271, 5:151, 6:9}#{0:35688*5, 1:325*5, 2:1118*5, 3:567*5, 4:271*5, 5:151*5, 6:9*5}
        lam = torch.from_numpy(np.random.beta(2.0, 2.0, (X.size(0), 1))).float()
        sn = torch.tensor([mapping[x] for x in label.tolist()]).unsqueeze(1).float()
        sn = 0.5*torch.sigmoid((sn-mu)/(beta*gamma))
        lam = sn*lam
        lam = lam.to(X.get_device())
        idx = list(range(X.size(0)))
        random.shuffle(idx)
        X = (1. - lam)*X + lam*X[idx]
        return X

    def forward(self, input_s, input_t, lens_s, lens_t, merge_idx=None, label=None):
        listened, lens_s_ = self.listener(input_s, lens_s)
        read, lens_t_ = self.reader(input_t, lens_t)
        mask_s, mask_t = get_mask(lens_s_.cpu().tolist()).to(read.get_device()), get_mask(lens_t_.cpu().tolist()).to(read.get_device())

        aligned_tq, attn_tq = self.attention(read.permute(1,0,2), listened.permute(1,0,2), mask_s.bool())

        self.rnn.flatten_parameters()
        out_tok_seq, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read.permute(1,0,2)], dim=2)))
        out_tok = extract(self.classifier_tok(self.dropout(out_tok_seq)).permute(1,0,2), mask_t.long())

        self.rnn_cls.flatten_parameters()
        out_feat, _ = self.rnn_cls(self.dropout(out_tok_seq[1:,:,:]))
        #out_cls = self.classifier_cls(merge(out_feat.permute(1,0,2), merge_idx))

        out_merge = merge(out_feat.permute(1,0,2), merge_idx)
        if label is not None:
            out_merge_ = self.mfw(out_merge, label)
            label_ = label
            #out_merge_, label_ = mixup2(out_merge, label, out_merge.get_device())
        else:
            out_merge_, label_ = out_merge, label
        out_cls = self.classifier_cls(out_merge_)

        return out_cls, out_tok, out_merge, attn_tq, label_

    def decouple(self, input_s, input_t, lens_s, lens_t, merge_idx=None, label_cat=None, label_det=None):
        listened, lens_s_ = self.listener(input_s, lens_s)
        read, lens_t_ = self.reader(input_t, lens_t)
        mask_s, mask_t = get_mask(lens_s_.cpu().tolist()).to(read.get_device()), get_mask(lens_t_.cpu().tolist()).to(read.get_device())

        aligned_tq, attn_tq = self.attention(read.permute(1,0,2), listened.permute(1,0,2), mask_s.bool())

        self.rnn.flatten_parameters()
        out_tok_seq, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read.permute(1,0,2)], dim=2)))
        out_tok = extract(self.classifier_tok(self.dropout(out_tok_seq)).permute(1,0,2), mask_t.long())

        self.rnn_cls.flatten_parameters()
        out_feat, _ = self.rnn_cls(self.dropout(out_tok_seq[1:,:,:]))
        #out_cls = self.classifier_cls(merge(out_feat.permute(1,0,2), merge_idx))

        out_merge = merge(out_feat.permute(1,0,2), merge_idx)
        if label_cat is not None:
            out_merge_cat = self.mfw(out_merge, label_cat, gamma=360.62002933219827, mu=205.75457241454885, beta=0.01)
        if label_det is not None:
            out_merge_det = self.mfw(out_merge, label_det, mapping={0:35688, 1:2441}, gamma=16623.5, mu=9333.509950709862, beta=0.01)

        out_cat = self.classifier_cls(out_merge_cat)
        out_det = self.classifier_det(out_merge_det)

        return out_cat, out_det, out_tok

    def decouple_hier(self, input_s, input_t, lens_s, lens_t, merge_idx=None, label_cat=None, label_det=None):
        listened, lens_s_ = self.listener(input_s, lens_s)
        read, lens_t_ = self.reader(input_t, lens_t)
        dev = read.get_device()
        if dev != -1:
            mask_s, mask_t = get_mask(lens_s_.cpu().tolist()).to(dev), get_mask(lens_t_.cpu().tolist()).to(dev)
        else:
            mask_s, mask_t = get_mask(lens_s_.cpu().tolist()), get_mask(lens_t_.cpu().tolist())

        aligned_tq, attn_tq = self.attention(read.permute(1,0,2), listened.permute(1,0,2), mask_s.bool())

        self.rnn.flatten_parameters()
        out_tok_seq, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read.permute(1,0,2)], dim=2)))
        out_tok = extract(self.classifier_tok(self.dropout(out_tok_seq)).permute(1,0,2), mask_t.long())

        out_merge, mask_new = merge_keep_seq(out_tok_seq[1:,:,:].permute(1,0,2), merge_idx)
        out_merge = out_merge.permute(1,0,2)

        self.rnn_cls.flatten_parameters()
        level_1_op, _ = self.rnn_cls(self.dropout(out_merge))

        level_1_op_unrav = extract(level_1_op.permute(1,0,2), mask_new.long())
        if label_det is not None:
            #level_1_op_unrav_ = self.mfw(level_1_op_unrav, label_det, mapping={0:36312, 1:4554}, gamma=15879.0, mu=12859.426425778096, beta=0.01)
            level_1_op_unrav_ = self.mfw(level_1_op_unrav, label_det, mapping={0:35688, 1:2441}, gamma=16623.5, mu=9333.509950709862, beta=0.01)
            out_det = self.classifier_det(level_1_op_unrav_)
        else:
            out_det = self.classifier_det(level_1_op_unrav)

        self.rnn_cls_2.flatten_parameters()
        level_2_op, _ = self.rnn_cls_2(self.dropout(level_1_op))

        level_2_op_unrav = extract(level_2_op.permute(1,0,2), mask_new.long())
        if label_cat is not None:
            #level_2_op_unrav_ = self.mfw(level_2_op_unrav, label_cat, gamma=1272.6455712412628, mu=680.0833416628293, beta=0.01)
            level_2_op_unrav_ = self.mfw(level_2_op_unrav, label_cat, gamma=12350.35015119583, mu=429.7652650998209, beta=0.01)
            out_cat = self.classifier_cls(level_2_op_unrav_)
        else:
            out_cat = self.classifier_cls(level_2_op_unrav)

        return out_cat, out_det, out_tok