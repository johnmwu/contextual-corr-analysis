import torch
from pytorch_transformers import XLNetTokenizer, XLNetModel, GPT2Tokenizer, GPT2Model, XLMTokenizer, XLMModel
import numpy as np 
import h5py

def get_mask(sentence, tokenizer, sep, model_name):

    ids = tokenizer.encode(sentence)

    #For each word, take the representation of its last sub-word
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    mask = np.full(len(segmented_tokens), False)

    if model_name.startswith('gpt2') or model_name.startswith('xlnet'):
        # if next token is a new word, take current token's representation
        #print(segmented_tokens)
        for i in range(len(segmented_tokens)-1):
            if segmented_tokens[i+1].startswith(sep):
                #print(i)
                mask[i] = True
        # always take the last token representation for the last word
        mask[-1] = True
    # example: ['jim</w>', 'henson</w>', 'was</w>', 'a</w>', 'pup', 'pe', 'teer</w>']
    elif model_name.startswith('xlm'):
        # if current token is a new word, take it
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].endswith(sep):
                mask[i] = True
        mask[-1] = True
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()	

    return mask, ids, segmented_tokens






xlnet_model_name = 'xlnet-base-cased'
xlm_model_name = 'xlm-mlm-ende-1024'
xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model_name)
xlm_tokenizer = XLMTokenizer.from_pretrained(xlm_model_name)
xlnet_sep = u'▁'
xlm_sep = '</w>'

#sentence = "Jim Henson is a pretty puppeteer"
sentence = "'' Mr. Allen objected to this analogy because it seems to `` assimilate the status of blacks to that of animals -- as a mere project of charity , of humaneness . ''"
print(sentence)
print(sentence.split())

xlnet_mask, xlnet_ids, xlnet_segmented_tokens = get_mask(sentence, xlnet_tokenizer, xlnet_sep, xlnet_model_name)
xlm_mask, xlm_ids, xlm_segmented_tokens = get_mask(sentence, xlm_tokenizer, xlm_sep, xlm_model_name)
xlnet_masked_ids = np.array(xlnet_ids)[xlnet_mask]
xlm_masked_ids = np.array(xlm_ids)[xlm_mask]
print('xlnet')
print(xlnet_tokenizer.tokenize(sentence))
print(xlnet_tokenizer.convert_tokens_to_ids(xlnet_tokenizer.tokenize(sentence)))
print(xlnet_tokenizer.encode(sentence))
print(xlnet_tokenizer.decode(xlnet_ids))
print('xlm')
print(xlm_tokenizer.tokenize(sentence))
print(xlm_tokenizer.convert_tokens_to_ids(xlm_tokenizer.tokenize(sentence)))
print(xlm_tokenizer.encode(sentence))
print(xlm_tokenizer.decode(xlm_ids))

print('compare')
print(xlnet_model_name, xlnet_mask, xlnet_ids, xlnet_segmented_tokens, xlnet_tokenizer.convert_ids_to_tokens(xlnet_masked_ids.tolist()), len(xlnet_masked_ids))
print(xlm_model_name, xlm_mask, xlm_ids, xlm_segmented_tokens, xlm_tokenizer.convert_ids_to_tokens(xlm_masked_ids.tolist()), len(xlm_masked_ids))

