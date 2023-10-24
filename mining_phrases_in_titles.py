import json
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

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


def title_candidates_extraction(title, text):
    
    stemmer = PorterStemmer()

    cans = extract_candidates(title)
    candidates = []
    for can in cans:
        candidates.append(can.lower())

    candidates = list(set(candidates))
    
    present_phrases = []
    absent_phrases = []
    text_low = text.lower()
    tokenized_text = meng17_tokenize(text_low)
    stem_text = [ stemmer.stem(word) for word in tokenized_text ]
    stem_text = ' '.join(stem_text)

    # stem_text = ' '.join(meng17_tokenize(text_low))
    # print(stem_text)


    for p in candidates:
        tokenized_p = meng17_tokenize(p.lower())
        stem_p = [ stemmer.stem(word) for word in tokenized_p ]
        stem_p = ' '.join(stem_p)
        # print(stem_p)
    
        if stem_p not in stem_text:
            absent_phrases.append(p)
        else:
            present_phrases.append(p)

    return present_phrases, absent_phrases


def mining_pres_abs_phrases_in_titles(dataset):

    processed_dataset = []

    for data in tqdm(dataset):
        temp = {}
        temp['title'] = data['title']
        title_present_phrase, title_absent_phrase = title_candidates_extraction(data['title'], data['abstract'])
        temp['abstract'] = data['abstract']
        temp['title_present_phrase'] = title_present_phrase
        temp['title_absent_phrase'] = title_absent_phrase
        temp['keyphrases'] = data['keywords']

        processed_dataset.append(temp)

    return processed_dataset


kp20k_train = read_jsonl('kp20k/train.json')
kp20k_train_mined = mining_pres_abs_phrases_in_titles(kp20k_train, 'kp20k')
save_as_jsonl(kp20k_train_mined, 'kp20k_title_phrases_mined.jsonl')
