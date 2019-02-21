import torch
from torch.autograd import Variable
import sys
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import *
import numpy as np
from collections import OrderedDict
import pickle
import json
import random



def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1

    sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)

    worddict = OrderedDict()
    for idx, word in enumerate(sorted_words):
        worddict[word] = idx+2 # 0: <eos>, 1: <unk>

    return worddict, wordcount


def convert_sentence_to_indices(sentence, worddict):

    indices = [
                  # assign an integer to each word, if the word is too rare assign unknown token
                  worddict.get(w) if worddict.get(w, VOCAB_SIZE + 1) < VOCAB_SIZE else UNK

                  for w in sentence.split()  # split into words on spaces
              ][: MAXLEN - 1]  # take only maxlen-1 words per sentence at the most.

    # last words are EOS
    indices += [EOS] * (MAXLEN - len(indices))

    indices = np.array(indices)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    indices = Variable(indices)
    if USE_CUDA:
        indices = indices.cuda()

    return indices


def encode(text, worddict, uniskip):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    ret = []

    for i,chunk in enumerate(chunks(text, 100)):
        print("encoding chunk {} of size {}".format(i, len(chunk)))
        indices = [convert_sentence_to_indices(sentence, worddict) for sentence in chunk]
        indices = torch.stack(indices)
        indices = uniskip(indices)
        indices = indices.view(-1, 2400)
        indices = indices.data.cpu().numpy()

        ret.extend(indices)
    ret = np.array(ret)

    return ret


def load_all_captions(data_dir,filenames):
    def load_captions(caption_name):  # self,
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions

    caption_dict = {}
    for key in filenames:
        caption_name = '%s/text/%s.txt' % (data_dir, key)
        captions = load_captions(caption_name)
        caption_dict[key] = captions
    return caption_dict


def load_json_file(data_dir):
    train = json.load(open(data_dir, 'r'))
    out = {}

    for i, img in enumerate(train):
        imgid = img['image_id']
        sents = img['caption']
        out[imgid] = np.array(sents)

    return out


def create_embedding(data_dir='./data/',SEED = 500, split_percent = 15):
    # Load the dataset and extract features
    captions = load_json_file(data_dir)
    # shuffle captions and devide to train/test
    c = list(captions.items())
    random.seed(SEED)
    random.shuffle(c)
    split = int(len(c) * split_percent / 100)
    test = dict(c[0:split])
    train = dict(c[split::])


    for name,dataset in enumerate([train,test]):
        if name == 0:
            name = 'train'
        else:
            name = 'test'
        text = np.array(list(dataset.values()))
        np.save('text-{}'.format(name), text)
        filenames = list(dataset.keys())
        text_shape = text.size

        worddict, wordcount = build_dictionary(text)

        vocab = list(worddict.keys())
        dir_st = 'data/skip-thoughts/'
        uniskip = UniSkip(dir_st, vocab)
        biskip = BiSkip(dir_st,vocab)
        if USE_CUDA:
            uniskip.cuda()
            biskip.cuda()

        features_uni = encode(text, worddict, uniskip)
        features_uni = np.reshape(features_uni,(text_shape,1,2400))
        features_bi = encode(text, worddict, biskip)
        features_bi = np.reshape(features_bi, (text_shape, 1, 2400))
        features = np.concatenate((features_uni, features_bi), axis=2)
        np.save('skip-thought-embeddings-{}'.format(name), features)
        with open('filenames-{}.pickle'.format(name), 'wb') as f:
            pickle.dump(filenames, f)



USE_CUDA = True
VOCAB_SIZE = 20000
MAXLEN = 30
EOS = 0  # to mean end of sentence
UNK = 1  # to mean unknown token

create_embedding(data_dir='data/oxford/vis_oxford.json',SEED = 500, split_percent = 15)
