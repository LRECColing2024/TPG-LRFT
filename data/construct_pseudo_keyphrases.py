import json
import nltk
import re
import json
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

import argparse

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_as_jsonl(data_list, file_path):
    """
    Save a list of dictionaries as a jsonl file.
    
    :param data_list: List of dictionaries.
    :param file_path: Path to the jsonl file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data_list:
            json_str = json.dumps(entry, ensure_ascii=False)
            file.write(json_str + '\n')


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


def extract_sub_candidates(text):
    GRAMMAR_EN = """  NP: {<NN.*|JJ>*<NN.*>}"""  # Adjective(s)(optional) + Noun(s)
    keyphrase_candidate = set()

    np_parser = nltk.RegexpParser(GRAMMAR_EN)  # Noun phrase parser
    tag = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    trees = np_parser.parse_sents(tag)  # Generator with one tree per sentence

    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))

            # Extract all sub-sequences of the NP (including sub-NPs)
            leaves = subtree.leaves()
            for i in range(len(leaves)):
                for j in range(i, len(leaves)):
                    sub_np = ' '.join(word for word, tag in leaves[i:j+1])
                    # Add to set if it's a valid NP (contains a noun)
                    if any(tag.startswith('NN') for _, tag in leaves[i:j+1]):
                        keyphrase_candidate.add(sub_np)

    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 4}

    return list(keyphrase_candidate)



def extract_candidates(text):

    GRAMMAR_EN = """  NP:
{<NN.*|JJ>*<NN.*>}"""   # Adjective(s)(optional) + Noun(s)
    keyphrase_candidate = set()
    

    np_parser = nltk.RegexpParser(GRAMMAR_EN)  # Noun phrase parser
    
    tag = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    
    trees = np_parser.parse_sents(tag)  # Generator with one tree per sentence
    #print(text)

    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))
    
    #print(keyphrase_candidate)
    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 4}
    #print(keyphrase_candidate)
  
    return list(keyphrase_candidate)

def text_candidates_extraction(text):
    #text_obj = InputTextObj(en_model, text, extract_sub=False)
    #cans = text_obj.keyphrase_candidate
    cans = extract_candidates(text)
    candidates = []
    for can, _ in cans:
        candidates.append(can.lower())

    candidates = list(set(candidates))

    return candidates

def sbert_calculate_similarity(text, phrases, model):
    text_embedding = model.encode(text)
    phrases_embeddings = model.encode(phrases)
   
    similarities =  [ util.dot_score(text_embedding, p_emb) for p_emb in phrases_embeddings ]
   
    sorted_phrases = [x for _, x in sorted(zip(similarities, phrases), reverse=True)]
    return sorted_phrases


processed_kp20k = read_jsonl('kp20k_title_phrases_mined.jsonl')

kp20k_abs_title_docs = []

for data in processed_kp20k:
    if len(data['title_absent_phrase']) >= 1:
        kp20k_abs_title_docs.append(data)

# print(len(kp20k_abs_title_docs))

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

stop_words = set(stopwords.words('english'))

present_added_kp20k = []


for data in tqdm(kp20k_abs_title_docs):
    temp = {}

    text = data['abstract']
    extracted_phrases = extract_candidates(text)

    extracted_ordered_phrases = sbert_calculate_similarity(text, extracted_phrases, model)
    extracted_filtered_phrases = [p for p in extracted_ordered_phrases if p.split()[0] not in stop_words and p.split()[-1] not in stop_words]

    title_present_phrases = data['title_present_phrase']
    stem_title_present_phrases = [ ' '.join(meng17_tokenize(tp)) for tp in title_present_phrases ]

    final_extracted_phrases = []
    for p in extracted_filtered_phrases:
        if ' '.join(meng17_tokenize(p)) not in title_present_phrases:
            final_extracted_phrases.append(p)

    # filter absent phrase
    title_sub_absent_phrases = []
    title_absent_phrases = data['title_absent_phrase']
    for tap in title_absent_phrases:
        sub_tap = extract_sub_candidates(tap)
        title_sub_absent_phrases.extend(sub_tap)
    ranked_sub_tap = sbert_calculate_similarity(data['abstract'], title_sub_absent_phrases, model)

    temp['title'] = data['title']
    temp['abstract'] = data['abstract']
    temp['title_present_phrases'] = data['title_present_phrase']
    temp['title_absent_phrases'] = data['title_absent_phrase']
    temp['ranked_sub_title_absent_phrases'] = ranked_sub_tap
    temp['abstract_present_phrases'] = final_extracted_phrases
    temp['gold_keyphrases'] = data['keyphrases']

    #print(temp)

    present_added_kp20k.append(temp)

save_as_jsonl(present_added_kp20k[:350000], 'kp20k_TPG_train.jsonl')
save_as_jsonl(present_added_kp20k[350000:], 'kp20k_TPG_valid.jsonl')
