import json
import pandas as pd

from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers.processors import TemplateProcessing
from nltk.stem import PorterStemmer
import re

def meng17_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits with <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', text)))

    return tokens

def TPG_LRFT_DataLoader(fname, tokenizer, batch_size, max_length, mode="train", stage='TPG'):
    """
    Build Data Loader

    """

    dataset = Dataset.from_json(fname)

    if not tokenizer.cls_token:
        tokenizer.cls_token = tokenizer.bos_token
    if not tokenizer.sep_token:
        tokenizer.sep_token = tokenizer.eos_token

    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{tokenizer.cls_token} $0 {tokenizer.sep_token}",
        pair=f"{tokenizer.cls_token} $A {tokenizer.sep_token} $B:1 {tokenizer.sep_token}:1",
        special_tokens=[(tokenizer.cls_token, tokenizer.cls_token_id), (tokenizer.sep_token, tokenizer.sep_token_id)],
    )

    

    def preprocess_function(examples):
    
        processed = {}

        if stage == 'TPG':
            abstract_present_phrases = examples['abstract_present_phrases']
            abstract_present_phrases = abstract_present_phrases[:10]

            masked_phrases = abstract_present_phrases[5:]  # masked in document
            not_masked_phrases = abstract_present_phrases[:5]

            masked_abstract = examples["abstract"]
            for phrase in masked_phrases:
                masked_abstract = re.sub(r'\b' + re.escape(phrase) + r'\b', '<mask>', masked_abstract, flags=re.IGNORECASE)

            input_text = f'{masked_abstract}'

        elif stage == 'LRFT':
            input_text = f'{examples["title"]}. {examples["abstract"]}'

        
        tokenizer_input = tokenizer(
            input_text,
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        processed["input_ids"] = tokenizer_input["input_ids"]
        processed["attention_mask"] = tokenizer_input["attention_mask"]

        if mode == "train":

            ps = PorterStemmer()

            preprocessed_text = meng17_tokenize(input_text)
            stem_text = ' '.join([ps.stem(w) for w in preprocessed_text])


            if stage == 'TPG':

                title_absent_phrases = examples['ranked_sub_title_absent_phrases'][:10]
                title_present_phrases = examples['title_present_phrases']

                silver_keyphrases_seq = title_present_phrases + not_masked_phrases + title_absent_phrases + masked_phrases 

                target_text = ';'.join(silver_keyphrases_seq)

            elif stage == 'LRFT':
                gold_keyphrases = examples['keywords']

                pres_keys = [p for p in gold_keyphrases if ' '.join([ps.stem(w) for w in meng17_tokenize(p)]) in stem_text]
                abs_keys = [p for p in gold_keyphrases if ' '.join([ps.stem(w) for w in meng17_tokenize(p)]) not in stem_text]

                pres_abs_keys = pres_keys + abs_keys

                target_text = ';'.join(pres_abs_keys)


            tokenizer_output = tokenizer(
                target_text,
                padding="max_length",
                max_length=256,
                truncation=True
            )
            processed["decoder_input_ids"] = tokenizer_output["input_ids"]
            processed["decoder_attention_mask"] = tokenizer_output["attention_mask"]

        return processed

    dataset = dataset.map(
        preprocess_function,
        num_proc=8,
        remove_columns=dataset.column_names
    ).with_format("torch")
    dataloader = DataLoader(dataset, shuffle=(True if mode=="train" else False), batch_size=batch_size, num_workers=8, pin_memory=True)

    return dataloader


def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]

    return j_list


def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')