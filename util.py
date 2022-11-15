import pickle
import torch.nn as nn
import torch
import math
import random
import re
import string
from transformers import BertTokenizer
from speechbrain.lobes.augment import SpecAugment
from speechbrain.processing.features import Deltas
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


TOK = BertTokenizer.from_pretrained("bert-base-uncased")
ASR_ID2TOK = ["<blank>"]+list(string.ascii_lowercase) + [str(x) for x in range(10)] + ["'","-","<space>"] + ["<tag>"]
ASR_TOK2ID = {x:i for i, x in enumerate(ASR_ID2TOK)}
TAG_TOK = "<tag>"


class FScore(nn.Module):
    def __init__(self, ncls=7, temp=1.0):
        super(FScore, self).__init__()
        self.temp = torch.tensor([[1.0, 0.1, 0.5, 0.2, 0.07, 0.05, 0.01]])
        self.ncls = 7

    def mhot_tgt(self, tlist):
        tgt = torch.zeros(len(tlist), self.ncls)
        for i,t in enumerate(tlist):
            tgt[i][t] = 1.
        return tgt

    def forward(self, pred, y_true): # pred --> (32, 7) y_true --> (32,)
        pred = torch.softmax(pred / self.temp.to(pred.get_device()), dim=1)
        pred = pred / torch.max(pred, dim=1, keepdim=True)[0]
        y_true_oh = self.mhot_tgt(y_true.cpu().tolist()).to(pred.get_device())
        TP = (pred*y_true_oh).sum(dim=0, keepdim=True)
        TPFP = pred.sum(dim=0, keepdim=True)
        Nk = y_true_oh.sum(dim=0, keepdim=True)
        return 1. - (2*TP/(TPFP + Nk + 1e-6)).mean()


class CBCrossEntropy(nn.Module):
    def __init__(self, ny, device, beta=0, gamma=0):
        super(CBCrossEntropy, self).__init__()  
        ny = torch.tensor(ny).unsqueeze(0).to(device) # (1,nclasses)
        self.W = (1. - beta) / (1. - (beta ** ny))
        self.gamma = gamma

    def forward(self, pred, gt):
        pred = -((1. - torch.softmax(pred, dim=1))**self.gamma) * (torch.log_softmax(pred, dim=1))
        pred = pred * self.W
        loss = torch.gather(pred, 1, gt.unsqueeze(1)).squeeze().mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, device, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, gt):
        pred = -((1. - torch.softmax(pred, dim=1))**self.gamma) * (torch.log_softmax(pred, dim=1))
        loss = torch.gather(pred, 1, gt.unsqueeze(1)).squeeze().mean()
        return loss


def list_batch(X, lens):
    sbatch_ = []
    for i, l in enumerate(lens):
        sbatch_.append(X[i,:l,:])
    return sbatch_


def pad_fac(input, factor=4):
    add_size = input.size(1) % factor
    if add_size != 0:
        rem_size = factor - add_size
        return torch.cat([input, torch.zeros(input.size(0), rem_size, input.size(2))], dim=1)
    else:
        return input


def SpecDel(X, lens, fmask, train=True):
    Del = Deltas(input_size=X.size(-1))
    X_ = list_batch(X, lens)
    xn = []
    for x in X_:
        x = x.unsqueeze(0)
        if train:
            SpecA = SpecAugment(time_warp=False, freq_mask_width=(0, fmask), time_mask_width=(0, int(0.05*x.size(1))), n_freq_mask=2, n_time_mask=10)
            x = SpecA(x)
        try:
            D = Del(x)
        except RuntimeError:
            D = Del(x)
        DD = Del(D)
        xn.append(torch.cat([x, D, DD], dim=2).squeeze(0))
    pack = pack_sequence(xn, enforce_sorted=False)
    Xn, _ = pad_packed_sequence(pack, batch_first=True)
    return Xn


def roll_in(x, lens):
    lens = [math.ceil(1.*k/4) for k in lens]
    x = pad_fac(x, 4)
    full = torch.cat([x, x.roll(-1, dims=1), x.roll(-2, dims=1), x.roll(-3, dims=1)], dim=2)
    extract = list(range(0, x.size(1), 4))
    return full[:, extract, :], lens


def convert_tok2id(text):
    lst = []
    for ch in text:
        if ch == " ":
            lst.append(ASR_TOK2ID["<space>"])
        else:
            lst.append(ASR_TOK2ID[ch])
    return lst


def convert_id2tok(ids):
    text = ''
    for x in ids:
        if x == ASR_TOK2ID["<space>"]:
            text = text + ' '
        else:
            text = text + ASR_ID2TOK[x]
    return text.replace("<blank>","")


def clean4asr(text):
    text = re.sub('[^A-Za-z0-9\s\-\']+','',text)
    return text.lower().strip()


def my_shuffle(x):
    if len(x) == 1:
        raise Exception
    for i in reversed(range(1, len(x))):
        # pick an element in x[:i] with which to exchange x[i]
        j = int(random.random() * i)
        x[i], x[j] = x[j], x[i]


def inject_seqn(X):
    if X.size(0) == 1:
        return X
    mask_sn = torch.ones(X.shape[0],1,1)
    zr = random.sample(list(range(X.shape[0])), int(0.2*X.shape[0]))
    mask_sn[zr] = 0.
    Z = list(range(X.shape[0]))
    my_shuffle(Z)
    X = torch.log(torch.exp(X) + 0.4 * torch.exp(X[Z]) * mask_sn)
    return X


def get_params(model):
    low = []
    high = []
    for name, parameter in model.named_parameters():
        if 'text_encoder' in name:
            low.append(parameter)
        else:
            high.append(parameter)
    return low, high


def load_dict(model, dict_path, loc='cuda:0', ddp=False):
    pretrained_dict = torch.load(dict_path, map_location=loc)
    model_dict = model.state_dict()
    new_pt_dict = {}
    for k, v in pretrained_dict.items():
        k_new = k
        if not ddp and k[:7] == 'module.':
            k_new = k[7:]
        elif ddp and k[:7] != 'module.':
            k_new = f'module.{k}'
        new_pt_dict[k_new] = v
    pretrained_dict = {k: v for k, v in new_pt_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def save_pick(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pick(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save(model, path):
    torch.save(model.state_dict(), path)


def load(model, path):
    model.load_state_dict(torch.load(path))


def save_checkpoint(state, filename):
    torch.save(state, filename)


def pt_load_dict(model, ptdict, ddp=True):
    #pretrained_dict = torch.load(dict_path, map_location=loc)
    pretrained_dict = ptdict
    model_dict = model.state_dict()
    new_pt_dict = {}
    for k, v in pretrained_dict.items():
        k_new = k
        if not ddp and k[:7] == 'module.':
            k_new = k[7:]
        elif ddp and k[:7] != 'module.':
            k_new = f'module.{k}'
        new_pt_dict[k_new] = v
    pretrained_dict = {k: v for k, v in new_pt_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model