import torch
from pytorch_transformers import XLNetTokenizer, XLNetModel, GPT2Tokenizer, GPT2Model, XLMTokenizer, XLMModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

import h5py
import json
import numpy as np
from tqdm import tqdm
import sys

from get_transformer_representations import get_sentences_from_hdf5, make_hdf5_file

disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def get_model_and_tokenizer(model_name, random_weights=False, do_basic_tokenize=False):

    if model_name.startswith('xlnet'):
        model = XLNetModel.from_pretrained(model_name, output_attentions=True).to(device)
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_basic_tokenize=do_basic_tokenize)    
        sep = u'▁'
    elif model_name.startswith('gpt2'):
        model = GPT2Model.from_pretrained(model_name, output_attentions=True).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, do_basic_tokenize=do_basic_tokenize)
        sep = 'Ġ'
    elif model_name.startswith('xlm'):
        model = XLMModel.from_pretrained(model_name, output_attentions=True).to(device)
        tokenizer = XLMTokenizer.from_pretrained(model_name, do_basic_tokenize=do_basic_tokenize)
        sep = '</w>'
    elif model_name.startswith('bert'):
        model = BertModel.from_pretrained(model_name, output_attentions=True).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_name, do_basic_tokenize=do_basic_tokenize)
        sep = '##'
    elif model_name.startswith('roberta'):
        model = RobertaModel.from_pretrained(model_name, output_attentions=True).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_basic_tokenize=do_basic_tokenize)
        sep = 'Ġ'        
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    if random_weights:
        model.apply(model.init_weights)

    return model, tokenizer, sep


# this follows the HuggingFace API for pytorch-transformers
def get_sentence_attn(sentence, model, tokenizer, sep, model_name):
    """
    Get attentions for one sentence

    return a numpy array of shape (num_layers, num_heads, sequence_length, sequence_length)
    """

    #print(sentence)

    with torch.no_grad():
        ids = tokenizer.encode(sentence)
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: list of torch.FloatTensor of shape (batch_size, num_heads, sequence_length, sequence_length) (attention weights after the attention softmax)
        all_attentions = model(input_ids)[-1]
        # squeeze batch dimension --> numpy array of shape (num_layers, num_heads, sequence_length, sequence_length)
        all_attentions = np.array([attention[0].cpu().numpy() for attention in all_attentions])


    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    #print(segmented_tokens)
    #print(all_attentions[0][0])
    #print(all_attentions.shape)
    assert len(segmented_tokens) == all_attentions.shape[2], 'incompatible tokens and states'

    # convert subword attention to word attention
    word_to_subword = get_word_to_subword(segmented_tokens, sep, model_name)
    all_attentions = [[get_word_word_attention(attention_h, word_to_subword) for attention_h in attention_l] for attention_l in all_attentions]
    all_attentions = np.array(all_attentions)

    return all_attentions


def get_word_to_subword(segmented_tokens, sep, model_name):
    """
    return a list of lists, where each element in the top list is a word and each nested list is indices of its subwords
    """

    word_to_subword = []

    # example: ['Jim', 'ĠHend', 'riks', 'Ġis', 'Ġa', 'Ġpupp', 'ete', 'er']
    # output: [[0], [1, 2], [3], [4], [5,6,7]]
    if model_name.startswith('gpt2') or model_name.startswith('xlnet') or model_name.startswith('roberta'):
        cur_word = [] 
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].startswith(sep):
                # don't append an empty list (may happen when i = 0)
                if len(cur_word) > 0:
                    word_to_subword.append(cur_word)
                cur_word = [i]
            else:
                cur_word.append(i)
        word_to_subword.append(cur_word)

    # example: ['j', 'im</w>', 'h', 'end', 'ri', 'ks</w>', 'is</w>', 'a</w>', 'pupp', 'et', 'eer</w>']
    # output: [[0, 1], [2, 3, 4, 5], [6], [7], [8,9,10]]
    elif model_name.startswith('xlm'):
        # TODO: implement these options 
        raise ValueError('not implemented model:', model_name)
        # if current token is a new word, take it
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].endswith(sep):
                mask[i] = True
        mask[-1] = True

    # example: ['Jim', 'He', '##nd', '##rik', '##s', 'is', 'a', 'puppet', '##eer']
    # output: [[0], [1,2,3,4], [5], [6], [7], [8,9]]
    elif model_name.startswith('bert'):
        cur_word = []
        for i in range(len(segmented_tokens)):
            if not segmented_tokens[i].startswith(sep):
                if len(cur_word) > 0:
                    word_to_subword.append(cur_word)
                cur_word = [i]
            else:
                cur_word.append(i)
        word_to_subword.append(cur_word)

    else:
        raise ValueError('Unrecognized model name:', model_name)

    return word_to_subword



# modified from Clark et al. 2019, What Does BERT Look At? An Analysis of BERT's Attention
def get_word_word_attention(token_token_attention, words_to_tokens, mode="mean"):
    """Convert token-token attention to word-word attention (when tokens are
    derived from words using something like byte-pair encodings)."""

    #print(token_token_attention)
    #print(words_to_tokens)

    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in words_to_tokens:
        not_word_starts += word[1:]

    # sum up the attentions for all tokens in a word that has been split
    for word in words_to_tokens:
        #print(word)
        word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

    # several options for combining attention maps for words that have been split
    # we use "mean" in the paper
    for word in words_to_tokens:
        if mode == "first":
            pass
        elif mode == "mean":
            word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
        elif mode == "max":
            word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
            word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
        else:
            raise ValueError("Unknown aggregation mode", mode)
    word_word_attention = np.delete(word_word_attention, not_word_starts, 0)

    return word_word_attention


# modified from https://github.com/nelson-liu/contextual-repr-analysis
def make_hdf5_file(sentence_to_index, vectors, output_file_path):
    with h5py.File(output_file_path, 'w') as fout:
        for key, embeddings in vectors.items():
            fout.create_dataset(
                str(key),
                embeddings.shape, dtype='float32',
                data=embeddings)
        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index",
            (1,),
            dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)


def run(input_hdf5_filename, model, tokenizer, sep, output_hdf5_filename, model_name):

    print('reading sentences from hdf5')
    hdf5 = h5py.File(input_hdf5_filename)
    sentence_to_idx = json.loads(hdf5['sentence_to_index'][0])
    hdf5.close()
    idx_to_attn = dict()

    print('getting attentions from model')
    for s, idx in tqdm(sentence_to_idx.items(), desc='attn'):
        attentions = get_sentence_attn(s, model, tokenizer, sep, model_name)
        idx_to_attn[idx] = attentions

    # TODO: verify that this works for the attentions
    print('writing attentions to new hdf5')
    make_hdf5_file(sentence_to_idx, idx_to_attn, output_hdf5_filename)


if __name__ == '__main__':
    random_weights = False
    if len(sys.argv) == 4:
        model_name = sys.argv[1]
        model, tokenizer, sep = get_model_and_tokenizer(model_name)
        input_hdf5_filename = sys.argv[2]
        output_hdf5_filename = sys.argv[3]
    elif len(sys.argv) == 5:
        model_name = sys.argv[1]
        random_weights = sys.argv[4].lower() == 'random'
        model, tokenizer, sep = get_model_and_tokenizer(model_name, random_weights=random_weights)
        
        input_hdf5_filename = sys.argv[2]
        output_hdf5_filename = sys.argv[3]
    else:
        print('USAGE: python ' + sys.argv[0] + ' <model name> <input hdf5 file> <output hdf5 file> [<random weights>]')
        print('pass <random weights> as "random" to generate attentions from randomly initialized models')

    run(input_hdf5_filename, model, tokenizer, sep, output_hdf5_filename, model_name)


