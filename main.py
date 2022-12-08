from transformers import pipeline
import gradio as gr
from model2 import *
from models import *
from tokenizers import Tokenizer
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torchaudio
import torchaudio.transforms as AT
import torch
import time
import os
from gtts import gTTS
import speech_recognition as sr
from jellyfish import soundex, levenshtein_distance, damerau_levenshtein_distance, hamming_distance, jaro_similarity
import pandas as pd
import numpy as np
import re
import datetime
from pydub import AudioSegment

# Model initializations
ID2CLS = {0: 'correct', 1: 'incorrect', 2: 'prompt', 3: 'repeat', 4: 'skip', 5: 'stutter', 6: 'tracking'}
ID2DET = {0: 'fluent', 1: 'not fluent'}
r = sr.Recognizer()
cntr = 1

# wav2vec2 for ASR
p = pipeline("automatic-speech-recognition")

# RNN-T for ASR (Reader)
rnnt_config = {'fmask': 27, 'nspeech-feat': 80, 'sample-rate': 16000, 'n_layer': 6,
              'in_dim': 960, 'hid_tr': 512, 'hid_pr': 1024, 'head_dim': 64,
              'nhead': 1, 'beam-size': 16, 'dropout': 0.25, 'enc_type': 'lstm',
              'deep_spec': True, 'unidirectional': True, 'vocab_size': len(ASR_ID2TOK),
              'ckpt_path': 'lstm_uni_adapt_readr0_tag.pth.tar', 'beam_size': 1}
compute_stft = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
compute_fbanks = Filterbank(n_mels=80)
rnnt_model = RNNT(rnnt_config)
checkpoint = torch.load(rnnt_config['ckpt_path'], map_location=f'cpu')
pt_load_dict(rnnt_model, checkpoint['state_dict'], ddp=False)
normalizer = checkpoint['normalizer'].to('cpu')
rnnt_model.eval()

# Detector for evaluation (Classifier)
tokenizer_path = "tokenizers/librispeech.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = 2 + tokenizer.get_vocab_size()
detector_config = {'embed_dim': 320, 'vocab_size': vocab_size, 'dropout': 0.1, 'input_dim': 80,
          'pyr_layer': 3, 'nlayer': 6, 'multi-gpu': False, 'nhead': 1, 'nclasses': 7, 'pretrain': False}
model_path = "deploy_readr.pt_best.pt"
norm_path = "readr.pickle"
detector_model = Detector(detector_config)
detector_model.eval()
load_dict(detector_model, model_path, loc=f'cpu')
norm = load_pick(norm_path)


def get_time(tml):
    """
        Gets time in seconds from the Gradio's stream state
        Parameters:
            tml - Gradio's stream state
        Returns:
            Time in seconds since the start of the speaking session
    """
    if len(tml) == 0:
        return 0
    elif len(tml) == 1:
        return 1
    else:
        sz = len(tml)
        return int(tml[sz-1]) - int(tml[0])


def track(trs, psg, st):
    """
        Tracks the words read in the reading passage
        Parameters:
            trs - Latest ASR Transcription
            psg - Complete Reading passage
            st - Gradio's stream state
        Returns:
            Index of the latest word read in the reading passage
    """
    trk = 0
    said = re.sub(' +', ' ', trs).lstrip().rstrip().split(" ")
    passage = psg.lstrip().rstrip().split(" ")
    
    # Evaluating indices & visted/tracked status of words in the reading passage to handle repeated words
    positions = []
    visited = {}
    for ix, elt in enumerate(passage):
        post = str(elt) + '_' + str(ix)
        if ix < st:
            visited[post] = 'v'
        else:
            visited[post] = 'nv'
        positions.append(post)

    # Evaluating ordered combinations of terms in the ASR transcription in case a single word is stretched 
    # across multiple terms (ca t ch i ld => catch ild or cat child etc.)
    if len(said) != 0:
        psgd = {k: v for v, k in enumerate(positions)}
        allc = []
        for i in range(len(said) + 1):
            for j in range(i + 1, min(i + 4, len(said) + 1)):
                allc.append("".join(said[i:j]))
        while '' in allc: allc.remove('')
        allw = positions
        if allc is not None:
            allw += allc
        
        # Evaluating soundex phonetic encodings
        report = pd.DataFrame([allw]).T
        report.columns = ['word']
        report[soundex.__name__] = report['word'].apply(lambda x: soundex(x.split('_')[0]))
        report2 = pd.DataFrame([allc]).T
        report2.columns = ['word']
        report.set_index('word', inplace=True)
        report2 = report.copy()
        report2[soundex.__name__] = np.nan

        # Evaluating the closest sounding words from the reading passage and the transcription 
        # in the current sliding window
        if allc is not None:
            for word in allc:
                closest_list = []
                for word_2 in positions:
                    if len(word_2.split('_')) == 2 and int(word_2.split('_')[1]) >= st:
                        if word != word_2.split('_')[0]:
                            closest = {}
                            closest['word'] = word_2
                            fw = report.loc[word, soundex.__name__]
                            sw = report.loc[word_2, soundex.__name__]
                            if isinstance(fw, pd.core.series.Series):
                                fw = fw.values[0]
                            if isinstance(sw, pd.core.series.Series):
                                sw = sw.values[0]
                            closest['similarity'] = levenshtein_distance(fw, sw)
                            closest_list.append(closest)
                if len(closest_list) > 0:
                    res = pd.DataFrame(closest_list).sort_values(by='similarity')
                    flg = False
                    for resix, reselt in res.iterrows():
                        for welt in reselt:
                            if welt in visited and visited[welt] == 'nv' and flg == False:
                                report2.loc[word, soundex.__name__] = welt
                                visited[welt] = 'v'
                                flg = True
                    fw = report2.loc[word, soundex.__name__]
                    if isinstance(fw, pd.core.series.Series):
                        fw = fw.values[0]
                    if psgd.get(fw):
                        trk = max(trk, psgd.get(fw))
    return trk


def transcribe(audio, state=""):
    """
        Transcribes the streaming audio input, tracks the read speech and invokes the Classifier
        Parameters:
            audio - Streaming audio input from microphone
            state - Gradio's stream state
        Returns:
            Invokes Classifier after ASR transcription & tracking
    """
    time.sleep(2)
    lines = ""
    with open("passage.txt", encoding='cp437') as f:
        lines = f.read()
    f.close()
    st_ix = 0
    read = " ".join(clean4asr(lines).split())
    if audio is None:
        state += '%#'
        return "", state, state, state
    till_list = re.findall('%.*?#', state)

    if len(till_list) >= 1:
        step_state = till_list[len(till_list) - 1].replace("%", "").replace("#", "")
    else:
        step_state = '%#'
    curr_state = "".join(";".join(re.findall('{.*?}', step_state)).replace("{", "")).replace("}", "").split(";")
    if len(curr_state) == 4:
        st_ix = max(int(curr_state[2]), int(st_ix))
        
    # ASR transcription from wav2vec2
    text = p(audio)["text"]
    timeT = str(int(time.time()))
    state += "%!" + timeT + "!"
    state += " <" + text + "> "
    readlist = read.lstrip().rstrip().split(" ")
    tr_state = " ".join(re.findall('<.*?>', state)).replace("<", "").replace(">", "")
    tm_state = " ".join(re.findall('!.*?!', state)).replace("!", "")
    tx_state = " ".join(re.findall('\$.*?\$', state)).replace("$", "")
    en_state = 0
    rt = re.findall('\$.*?\#', state)
    for rte in rt:
        pt = rte.replace("\$", "").replace("#", "")
        ptl = re.findall('{.*?}', pt)
        en_state = max(en_state, int(ptl[2].replace('{', '').replace('}','')))
    timed = get_time(tm_state)
    if timed%4 == 0:
        # ASR transcription from RNN-T Reader (run every 4th second for lengthier audio input chunks)
        hyph = get_asr1(audio)
        state += " $" + hyph + "$ "
    else:
        state += " $$ "
    
    # Tracking the index of the latest read word using the transcriptions
    en_iy = track(tx_state, read, st_ix)
    en_ix = max(max(en_iy, track(tr_state, read, st_ix) + 1), en_state)
    till = " ".join(readlist[: en_ix])
    rest = readlist[en_ix :]
    em = 0
    
    # Adding prompts in case a <tag> is produced in the ASR transcription
    for iy, yts in enumerate(reversed(tx_state.split(" "))):
        if yts == '<tag>':
                return get_report_teach(till, audio, st_ix, en_ix, state, rest, yts[min(iy+1, len(tx_state.split(" "))-1)])

    # Adding prompts in case of silence (hesitations/pauses)
    for yts in reversed(tr_state):
        if yts == ' ':
            em = em + 1
            global cntr
            if em >= 2 * cntr:
                cntr = cntr + 1
                return get_report_teach(till, audio, st_ix, en_ix, state, rest, " ".join(readlist[max(0, en_ix-1): en_ix+1]))
        else:
            break
    return get_report_teach(till, audio, st_ix, en_ix, state, rest, "")


def transcribe4(audio, state=""):
    """
        Transcribes the streaming audio input, tracks the read speech and invokes the Classifier
        Parameters:
            audio - Streaming audio input from microphone
            state - Gradio's stream state
        Returns:
            Invokes Classifier after ASR transcription & tracking
    """
    time.sleep(4)
    lines = ""
    with open("passage.txt", encoding='cp437') as f:
        lines = f.read()
    f.close()
    st_ix = 0
    read = " ".join(clean4asr(lines).split())
    if audio is None:
        state += '%#'
        return "", state, state, state
    till_list = re.findall('%.*?#', state)

    if len(till_list) >= 1:
        step_state = till_list[len(till_list) - 1].replace("%", "").replace("#", "")
    else:
        step_state = '%#'
    curr_state = "".join(";".join(re.findall('{.*?}', step_state)).replace("{", "")).replace("}", "").split(";")
    if len(curr_state) == 4:
        st_ix = max(int(curr_state[2]), int(st_ix))
    
    # ASR transcription from RNN-T Reader every 4th second
    text = get_asr1(audio)
    
    timeT = str(int(time.time()))
    state += "%!" + timeT + "!"
    state += " <" + text + "> "
    readlist = read.lstrip().rstrip().split(" ")
    yt_state = " ".join(re.findall('<.*?>', state)).replace("<", "").replace(">", "")
    tm_state = " ".join(re.findall('!.*?!', state)).replace("!", "")
    tx_state = " ".join(re.findall('\$.*?\$', state)).replace("$", "")
    en_state = 0
    rt = re.findall('\$.*?\#', state)
    for rte in rt:
        pt = rte.replace("\$", "").replace("#", "")
        ptl = re.findall('{.*?}', pt)
        en_state = max(en_state, int(ptl[2].replace('{', '').replace('}', '')))
    timed = get_time(tm_state)
    if timed % 4 == 0:
        hyph = get_asr1(audio)
        state += " $" + hyph + "$ "
    else:
        state += " $$ "
    
    # Tracking the index of the latest read word using the transcription
    en_iy = track(tx_state, read, st_ix)
    en_ix = max(en_iy, en_state)
    till = " ".join(readlist[: en_ix])
    rest = readlist[en_ix:]
    em = 0
    
    # Adding prompts in case a <tag> is detected
    for iy, yts in enumerate(reversed(tx_state.split(" "))):
        if yts == '<tag>':
            return get_report_teach(till, audio, st_ix, en_ix, state, rest,
                                    yts[min(iy + 1, len(tx_state.split(" ")) - 1)])

    # Adding prompts in case of silence (hesitations)
    for yts in reversed(yt_state):
        if yts == ' ':
            em = em + 1
            global cntr
            if em >= 2 * cntr:
                cntr = cntr + 1
                return get_report_teach(till, audio, st_ix, en_ix, state, rest,
                                        " ".join(readlist[max(0, en_ix - 1): en_ix + 1]))
        else:
            break
    return get_report_teach(till, audio, st_ix, en_ix, state, rest, "")


def get_report_teach(read, audio, st_ix, en_ix, state, full, prm=""):
    """
        Detects & classifies disfluencies in the reading and provides visual (color highlights) & audio (teacher prompts)
        Parameters:
            read - Portion of the reading passage read till now (from starting word till tracked word)
            audio - Streaming audio input from microphone
            st_ix - Index of the starting word in the reading passage read in the current sliding window
            en_ix - Index of the ending/tracked word in the reading passage read in the current sliding window
            state - Gradio's stream state
            full - Complete reading passage
            prm - Word(s) to be prompted
        Returns:
            List of tuples consisting of words from the reading passage & the disfluent classes they fall into
            Auto played hidden audio element to provide oral prompts
            Gradio's stream state
    """
    if audio is None:
        return "", "", "", ""
    all_res = []
    cls_res = []
    cls_res_scr = []
    text = clean4asr(read)
    words = text.split()
    y_cls = []
    y_scr = []
    
    # Classification into disfluent classes & evaluation of current score
    if len(read) > 0:
        wav, org_sr = torchaudio.load(audio)
        wav = AT.Resample(org_sr, 16000)(wav)
        features = compute_stft(wav)
        features = spectral_magnitude(features)
        features = compute_fbanks(features)
        speech_batch = [pad(features.squeeze(0), factor=8).unsqueeze(0)]
        X, lens, lmax = padding(speech_batch)
        lens_s = torch.tensor(lens)
        text_raw = [text]
        tokenized = tokenizer.encode_batch(text_raw)
        text_batch = pack_sequence([torch.tensor([30000] + x.ids).long() for x in tokenized], enforce_sorted=False)
        text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)
        merge_idx = [merge(x.tokens) for x in tokenized]
        lens_norm = lens_s / lmax
        sbatch = norm(X, lens_norm.float(), epoch=1000)
        with torch.no_grad():
            pred_cat, pred_det, _ = detector_model.decouple_hier(sbatch, text_batch, lens_s, lens_t, merge_idx)
        y_det = torch.max(pred_det, dim=1)[1].cpu().tolist()
        y_det = [ID2DET[x] for x in y_det]
        scr_res = 0
        for i, detection in enumerate(y_det):
            if detection == 'not fluent':
                y_cls.append(ID2CLS[torch.argmax(pred_cat[i][1:]).item() + 1])
                y_scr.append(scr_res)
            else:
                y_cls.append(ID2CLS[0])
                scr_res = scr_res + 1
                y_scr.append(scr_res)
        cls_res = zip(words[st_ix: en_ix], y_cls[st_ix: en_ix])
        cls_res_scr = list(zip(words, y_cls, y_scr))
    cls_res_str = ""
    prmpt = ''
    fprt = ''

    sprt = ''
    filen = "silent.wav"

    if prm != '':
        fprt = prm

    flg = False
    for wrd, wrd_res in cls_res:
        cls_res_str += " [" + wrd + " : " + wrd_res + "], "
        if wrd_res != 'correct' and flg == False:
            sprt = wrd
            flg = True

    dts = str(datetime.datetime.now())\
        .replace("-", "")\
        .replace(" ", "")\
        .replace(":", "")\
        .replace(".", "")
    for fils in os.listdir():
        if re.search('welcome*', fils):
            os.remove(fils)
  
    # Synthesizing teacher prompts - in case of long hesitations & disfluency detection
    if sprt != '':
        # In case of long 
        prmpt = sprt
        filen = "welcome" + dts + ".mp3"
    elif sprt == '' and fprt != '':
        prmpt = fprt
        filen = "welcome" + dts + ".mp3"
        aud = gTTS(text=clean4asr(prmpt), lang='en', slow=True).save(filen)
        aud = AudioSegment.from_mp3(filen)
        aud = aud._spawn(aud.raw_data, overrides={"frame_rate": int(aud.frame_rate * 0.75)})
        aud.set_frame_rate(aud.frame_rate).export(filen)
    else:
        prmpt = 'read'
    
    # Adding current classifications, sliding window updates and scores to Gradio's stream state
    fullz = list(zip(full, ["yet to read"] * len(full)))
    firstz = list(zip(words, y_cls))
    firstz.extend(fullz)
    state += '{' + cls_res_str + '},{' + str(st_ix) + '},{' + str(en_ix) + '},{' + prmpt + '}# '
    disp_res_list = ";".join(re.findall('>.*?#', state)).replace('>', '').replace('#', '').split(';')
    scr_res = 0
    for dl_str in disp_res_list:
        el_str = re.findall('{.*?}', dl_str)
        el_str_c = el_str[0].replace('{', '').replace('}', '')
        if len(el_str_c) > 1:
            el_str_c_lt = el_str_c.split(',')
            for elt_c in el_str_c_lt:
                elt_cln = elt_c.lstrip().rstrip()
                if len(elt_cln) > 1:
                    sbelt = elt_cln.split(':')
                    el_wrd = sbelt[0].replace(']', '').replace('[', '').lstrip().rstrip()
                    el_cls = sbelt[1].replace(']', '').replace('[', '').lstrip().rstrip()
                    if el_cls == 'correct':
                        scr_res = scr_res + 1
                    all_res.append((el_wrd, el_cls, scr_res))
                    
    htmlText = f"""
        <div>
            <br>
            <h2 style="text-align:center">You got <b>{max(0, len(cls_res_scr))}</b> out of <b>{len(firstz)}</b> words 
            right!</h2> </div> """
    htmlText += f"""
            <div style="display:none">
            <audio id="tp" class="w-full h-14 p-2 mt-7" controls="" preload="metadata" src='file/{filen}' autoplay>
            </audio>
            </div>
            """
    return firstz, htmlText, state


def get_report_file(filen, audio, state=""):
    """
        Invokes the Classifier to classify & evaluate the read speech in an uploaded audio file using the text from a
        reading passage into disfluency classes, produces visual feedback using colorful highlighted text
        Parameters:
            filen - Text file containing the reading passage
            audio - Audio input from an uploaded audio file
            state - Gradio's stream state
        Returns:
            List of tuples consisting of words from the reading passage & the disfluent classes they fall into
            HTMLText showing count of words read correctly
            Gradio's stream state
    """
    filen.seek(0)
    read = filen.read().decode('utf-8')
    if audio is None:
        return "", ""
    
    # Classification into disfluent classes & evaluation of current score
    wav, org_sr = torchaudio.load(audio)
    wav = AT.Resample(org_sr, 16000)(wav)
    features = compute_stft(wav)
    features = spectral_magnitude(features)
    features = compute_fbanks(features)
    text = clean4asr(read)
    speech_batch = [pad(features.squeeze(0), factor=8).unsqueeze(0)]
    X, lens, lmax = padding(speech_batch)
    lens_s = torch.tensor(lens)
    text_raw = [text]
    tokenized = tokenizer.encode_batch(text_raw)
    text_batch = pack_sequence([torch.tensor([30000] + x.ids).long() for x in tokenized], enforce_sorted=False)
    text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)
    merge_idx = [merge(x.tokens) for x in tokenized]
    lens_norm = lens_s / lmax
    sbatch = norm(X, lens_norm.float(), epoch=1000)
    with torch.no_grad():
        pred_cat, pred_det, _ = detector_model.decouple_hier(sbatch, text_batch, lens_s, lens_t, merge_idx)
    y_det = torch.max(pred_det, dim=1)[1].cpu().tolist()
    y_det = [ID2DET[x] for x in y_det]
    y_cls = []
    y_scr = []
    scr = 0
    for i, detection in enumerate(y_det):
        if detection == 'not fluent':
            y_cls.append(ID2CLS[torch.argmax(pred_cat[i][1:]).item() + 1])
            y_scr.append(scr)
        else:
            y_cls.append(ID2CLS[0])
            scr = scr + 1
            y_scr.append(scr)
    words = text.split()
    state += str(y_cls)
    htmlText = f"""
        <div>
        <br>
        <h2 style="text-align:center">You got <b>{y_scr[len(y_scr)-1]}</b> out of <b>{len(y_det)}</b> words right!</h2>
        </div>
        """
    return list(zip(words, y_cls)), htmlText, state


def get_asr1(audio):
    """
       Transcribes the streaming audio input using the RNN-T Reader model
       Parameters:
           audio - Streaming audio input from a microphone
       Returns:
           ASR transcription of the audio
    """
    wav, org_sr = torchaudio.load(audio)
    wav = AT.Resample(org_sr, 16000)(wav)
    features = compute_stft(wav)
    features = spectral_magnitude(features)
    features = compute_fbanks(features)
    speechL = [features.squeeze(0)]
    pack1 = pack_sequence(speechL, enforce_sorted=False)
    speechB, logitLens = pad_packed_sequence(pack1, batch_first=True)
    lens_norm = [1.]
    speechB = normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000)
    speechB = SpecDel(speechB, logitLens, rnnt_config['fmask'], train=False)
    speechB, logitLens = roll_in(speechB, logitLens)
    hyp, score = rnnt_model.beam_search(speechB, beam_size=rnnt_config['beam_size'])
    hypText = convert_id2tok(hyp)
    return hypText


def get_asr(audio):
    """
       Transcribes the audio input using the RNN-T Reader model
       Parameters:
           audio - Audio input from an uploaded audio file
       Returns:
           ASR transcription of the audio
    """
    wav, org_sr = torchaudio.load(audio)
    wav = AT.Resample(org_sr, 16000)(wav)
    features = compute_stft(wav)
    features = spectral_magnitude(features)
    features = compute_fbanks(features)
    speechL = [features.squeeze(0)]
    pack1 = pack_sequence(speechL, enforce_sorted=False)
    speechB, logitLens = pad_packed_sequence(pack1, batch_first=True)
    lens_norm = [1.]
    speechB = normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000)
    speechB = SpecDel(speechB, logitLens, rnnt_config['fmask'], train=False)
    speechB, logitLens = roll_in(speechB, logitLens)
    hyp, score = rnnt_model.beam_search(speechB, beam_size=rnnt_config['beam_size'])
    hypText = convert_id2tok(hyp)
    htmlText = f"""
                    <div>
                    <h1 style="font-size:30px;line-height:1.5em;">{hypText}</h1>
                    </div>
                    """
    return htmlText


def speak(filen):
    """
       Produces text-to-speech output using a synthesized teacher's voice
       Parameters:
           filen - Text file with the passage to be read
       Returns:
           Audio of the synthesized speech
    """
    filen.seek(0)
    lines = filen.read().decode('utf-8')
    aud = gTTS(text=clean4asr(lines), lang='en', slow=True).save("welcome.mp3")
    aud = AudioSegment.from_mp3("welcome.mp3")
    aud = aud._spawn(aud.raw_data, overrides={"frame_rate": int(aud.frame_rate * 0.75)})
    aud.set_frame_rate(aud.frame_rate).export("welcome.mp3")
    htmlText = f"""
                <div>
                <h1 style="font-size:30px;line-height:1.5em;">{lines}</h1>
                </div>
                <div style="display:none">
                <audio id="tp" class="w-full h-14 p-2 mt-7" controls="" preload="metadata" src='file/welcome.mp3' autoplay >
                </audio>
                </div>
                """
    return htmlText


# Tab 1 provides a slowed down reading (synthesized teacher's voice) of a txt file containing the reading passage
tab1 = gr.Interface(
    fn=speak,
    inputs=[gr.File(type="file", file_count="single", label="Reading passage")],
    outputs=[gr.HTML(label="Audio")],
    live=True,
    css="body {background-color: red}",
    allow_flagging="never",
    show_error=False,
    show_api=False
)

# Tab 2 provides visual (color highlights) and oral (synthesised teacher prompts) feedback in real-time
# For the ASR transcription, it uses the wav2vec2 and RNN-T Reader; takes as input streaming audio from the microphone
tab2 = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=True).style(rounded=True),
            "state"],
    outputs=[gr.HighlightedText(label="Reading", color_map={"correct": "rgb(0,255,0)", "tracking": "rgb(0,191,255)", "incorrect": "rgb(255,0,0)", "stutter": "rgb(255,255,0)", "skip": "rgb(148,0,211)", "prompt": "rgb(255,165,0)", "repeat": "rgb(255,105,180)", "yet to read" : "rgb(220,220,220)"}, show_legend=True, elem_id="ht1"),
             gr.outputs.HTML(label="You read"),
             "state"],
    live=True,
    allow_flagging="never",
    show_error=False,
    show_api=False,
    css='body{background-color:white;}',
    theme="grass"
)

# Tab 3 provides visual (color highlights) and oral (synthesised teacher prompts) feedback in real-time
# For the ASR transcription, it uses the RNN-T Reader; takes as input streaming audio from the microphone
tab3 = gr.Interface(
    fn=transcribe4,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=True),
            "state"],
    outputs=[gr.HighlightedText(label="Result", color_map={"correct": "rgb(0,255,0)", "tracking": "rgb(0,191,255)", "incorrect": "rgb(255,0,0)", "stutter": "rgb(255,255,0)", "skip": "rgb(148,0,211)", "prompt": "rgb(255,165,0)", "repeat": "rgb(255,105,180)", "yet to read" : "rgb(220,220,220)"}, show_legend=True),
        gr.outputs.HTML(), "state"],
    live=True,
    allow_flagging="never",
    show_error=False,
    show_api=False
)

# Tab 4 runs the Classifier on a complete uploaded audio file & a text file containing the reading passage to classify
# the uttered words in the passage into the various disfluency classes
tab4 = gr.Interface(
    fn=get_report_file,
    inputs=[gr.File(type="file", file_count="single", label="Reading passage"),
            gr.Audio(source="upload", type="filepath", label="Audio File"), "state"],
    outputs=[gr.HighlightedText(label="Result", color_map={"correct": "rgb(0,255,0)", "tracking": "rgb(0,191,255)", "incorrect": "rgb(255,0,0)", "stutter": "rgb(255,255,0)", "skip": "rgb(148,0,211)", "prompt": "rgb(255,165,0)", "repeat": "rgb(255,105,180)", "yet to read" : "rgb(220,220,220)"}, show_legend=True),
             gr.HTML(),
             "state"],
    live=True,
    allow_flagging="never",
    show_error=False,
    show_api=False
)

# Tab 5 runs the RNN-T Reader on a complete uploaded audio file to provide the ASR transcription
tab5 = gr.Interface(
    fn=get_asr,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.HTML(label="You said"),
    live=True,
    allow_flagging="never",
    show_error=False,
    show_api=False
)

# Gradio's tabbed interface element to group all the tabs in a single demo
demo = gr.TabbedInterface(interface_list=[tab1, tab2, tab3, tab4, tab5],
                          tab_names=["Listen to a Teacher", "Read with a Teacher 1",
                                     "Read with a Teacher 2", "Upload a recorded file", "Transcribe a file"])


if __name__ == "__main__":
    demo.launch(debug=True)
