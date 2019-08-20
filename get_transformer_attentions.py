import torch
from pytorch_transformers import XLNetTokenizer, XLNetModel, GPT2Tokenizer, GPT2Model, XLMTokenizer, XLMModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

import h5py
import json
import numpy as np
from tqdm import tqdm
import sys

from get_transformer_representations import get_model_and_tokenizer

disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# this follows the HuggingFace API for pytorch-transformers
def get_sentence_repr(sentence, model, tokenizer, sep, model_name):
    """
    Get representations for one sentence
    """

    with torch.no_grad():
        ids = tokenizer.encode(sentence)
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: list of torch.FloatTensor of shape (batch_size, num_heads, sequence_length, sequence_length) (attention weights after the attention softmax)
        all_attentions = model(input_ids)[-1]
        # squeeze batch dimension --> numpy array of shape (num_layers, num_heads, sequence_length, sequence_length)
        all_attentions = np.array([attention[0].cpu().numpy() for attention in all_attentions])


    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert len(segmented_tokens) == all_attentions.shape[2], 'incompatible tokens and states'

    # convert subword attention to word attention
    word_to_subword = get_word_to_subword(segmented_tokens, sep, model_name)
    all_attentions = [[[get_word_word_attention(attention, words_to_tokens) for attention in attention_h] for attention_h in attention_l] for attention_l in all_attentions]





        # convert to format required for contexteval: numpy array of shape (num_layers, sequence_length, representation_dim)
        all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states[:-1]]
        all_hidden_states = np.array(all_hidden_states)



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
        # if current token is a new word, take it
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].endswith(sep):
                mask[i] = True
        mask[-1] = True
    # example: ['Jim', 'He', '##nd', '##rik', '##s', 'is', 'a', 'puppet', '##eer']
    # output: [[0], [1,2,3,4], [5], [6], [7], [8,9]]
    elif model_name.startswith('bert'):
        # if next token is not a continuation, take current token's representation
        for i in range(len(segmented_tokens)-1):
            if not segmented_tokens[i+1].startswith(sep):
                mask[i] = True
        mask[-1] = True
    else:
        raise ValueError('Unrecognized model name:', model_name)

    return word_to_subword



# modified from Clark et al. 2019, What Does BERT Look At? An Analysis of BERT's Attention
def get_word_word_attention(token_token_attention, words_to_tokens, mode="mean"):
    """Convert token-token attention to word-word attention (when tokens are
    derived from words using something like byte-pair encodings)."""

    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in words_to_tokens:
        not_word_starts += word[1:]

    # sum up the attentions for all tokens in a word that has been split
    for word in words_to_tokens:
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