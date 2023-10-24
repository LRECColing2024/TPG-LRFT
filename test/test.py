import re
import json
import torch
import argparse
from nltk.stem import PorterStemmer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

from transformers import BartTokenizer, BartForConditionalGeneration

parser = argparse.ArgumentParser(description='Script converted from notebook.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model.')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Test Dataset.')
parser.add_argument('--num_beams', type=int, default=20, help='beam size for beam search.')

args = parser.parse_args()

model_path = args.model_path
dataset_path = args.dataset_path
num_beams = args.num_beams

with open(dataset_path, 'r') as f:
    data = [json.loads(line) for line in f]

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained(model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model.to(device)


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


def precision_recall_f1(predicted, labels, k):

    ps = PorterStemmer()

    predicted = predicted[:k]

    predicted_stem = set([' '.join([ps.stem(w) for w in meng17_tokenize(p)]) for p in predicted])
    labels_stem = set([' '.join([ps.stem(w) for w in meng17_tokenize(label)]) for label in labels])

    tp = len(labels_stem & predicted_stem)

    precision = tp / len(predicted_stem) if len(predicted_stem) > 0 else 0.0
    recall = tp / len(labels_stem) if len(labels_stem) > 0 else 0.0
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
    

def total_evaluate(preds, labels):

    precision_3_p, recall_3_p, f1_3_p = 0, 0, 0
    precision_5_p, recall_5_p, f1_5_p = 0, 0, 0
    precision_10_p, recall_10_p, f1_10_p = 0, 0, 0
    precision_20_p, recall_20_p, f1_20_p = 0, 0, 0
    precision_O_p, recall_O_p, f1_O_p = 0, 0, 0,
    precision_M_p, recall_M_p, f1_M_p = 0, 0, 0,


    f1_3_list, precision_3_list, recall_3_list = [], [], []
    f1_5_list, precision_5_list, recall_5_list = [], [], []
    f1_10_list, precision_10_list, recall_10_list = [], [], []
    f1_20_list, precision_20_list, recall_20_list = [], [], []
    f1_O_list, precision_O_list, recall_O_list = [], [], []
    f1_M_list, precision_M_list, recall_M_list = [], [], []

    num_of_samples = 0

    for pred, label in zip(preds, labels):

        p_3_p, r_3_p, f1_score_3_p = precision_recall_f1(pred, label, 3)
        precision_3_p += p_3_p
        recall_3_p += r_3_p
        f1_3_p += f1_score_3_p

        p_5_p, r_5_p, f1_score_5_p = precision_recall_f1(pred, label, 5)
        precision_5_p += p_5_p
        recall_5_p += r_5_p
        f1_5_p += f1_score_5_p

        p_10_p, r_10_p, f1_score_10_p = precision_recall_f1(pred, label, 10)
        precision_10_p += p_10_p
        recall_10_p += r_10_p
        f1_10_p += f1_score_10_p

        p_20_p, r_20_p, f1_score_20_p = precision_recall_f1(pred, label, 20)
        precision_20_p += p_20_p
        recall_20_p += r_20_p
        f1_20_p += f1_score_20_p

        p_O_p, r_O_p, f1_score_O_p = precision_recall_f1(pred, label, len(label))
        precision_O_p += p_O_p
        recall_O_p += r_O_p
        f1_O_p += f1_score_O_p

        p_M_p, r_M_p, f1_score_M_p = precision_recall_f1(pred, label, len(pred))
        precision_M_p += p_M_p
        recall_M_p += r_M_p
        f1_M_p += f1_score_M_p

        if len(label) != 0:
            
            f1_3_list.append(f1_score_3_p)
            precision_3_list.append(p_3_p)
            recall_3_list.append(r_3_p)

            f1_5_list.append(f1_score_5_p)
            precision_5_list.append(p_5_p)
            recall_5_list.append(r_5_p)

            f1_10_list.append(f1_score_10_p)
            precision_10_list.append(p_10_p)
            recall_10_list.append(r_10_p)

            f1_20_list.append(f1_score_20_p)
            precision_20_list.append(p_20_p)
            recall_20_list.append(r_20_p)

            f1_O_list.append(f1_score_O_p)
            precision_O_list.append(p_O_p)
            recall_O_list.append(r_O_p)

            f1_M_list.append(f1_score_M_p)
            precision_M_list.append(p_M_p)
            recall_M_list.append(r_M_p)

            num_of_samples += 1  

    
    precision_3_p /= num_of_samples
    recall_3_p /= num_of_samples
    f1_3_p /= num_of_samples

    precision_5_p /= num_of_samples
    recall_5_p /= num_of_samples
    f1_5_p /= num_of_samples

    precision_10_p /= num_of_samples
    recall_10_p /= num_of_samples
    f1_10_p /= num_of_samples

    precision_20_p /= num_of_samples
    recall_20_p /= num_of_samples
    f1_20_p /= num_of_samples

    precision_O_p /= num_of_samples
    recall_O_p /= num_of_samples
    f1_O_p /= num_of_samples

    precision_M_p /= num_of_samples
    recall_M_p /= num_of_samples
    f1_M_p /= num_of_samples


    print('F1@3: {:.4f}, Precision@3: {:.4f}, Recall@3: {:.4f}'.format(f1_3_p, precision_3_p, recall_3_p))
    print('F1@5: {:.4f}, Precision@5: {:.4f}, Recall@5: {:.4f}'.format(f1_5_p, precision_5_p, recall_5_p))
    print('F1@10: {:.4f}, Precision@10: {:.4f}, Recall@10: {:.4f}'.format(f1_10_p, precision_10_p, recall_10_p))
    print('F1@20: {:.4f}, Precision@20: {:.4f}, Recall@20: {:.4f}'.format(f1_20_p, precision_20_p, recall_20_p))
    print('F1@O: {:.4f}, Precision@O: {:.4f}, Recall@O: {:.4f}'.format(f1_O_p, precision_O_p, recall_O_p))
    print('F1@M: {:.4f}, Precision@M: {:.4f}, Recall@M: {:.4f}'.format(f1_M_p, precision_M_p, recall_M_p))

    
    results = {
        'F1@3_mean': f1_3_p, 
        'Precision@3_mean': precision_3_p,
        'Recall@3_mean': recall_3_p,
        
        'F1@5_mean': f1_5_p,
        'Precision@5_mean': precision_5_p, 
        'Recall@5_mean': recall_5_p,
        
        'F1@10_mean': f1_10_p, 
        'Precision@10_mean': precision_10_p,
        'Recall@10_mean': recall_10_p,

        'F1@20_mean': f1_20_p,
        'Precision@20_mean': precision_20_p, 
        'Recall@20_mean': recall_20_p, 
        
        'F1@O_mean': f1_O_p, 
        'Precision@O_mean': precision_O_p, 
        'Recall@O_mean': recall_O_p, 

        'F1@M_mean': f1_M_p, 
        'Precision@M_mean': precision_M_p, 
        'Recall@M_mean': recall_M_p
    }

    return results



def generate_keyphrases(model, text, tokenizer, device=device, max_length=512, num_beams=20):
    
    model.eval()  
    model.to(device)  

    with torch.no_grad():
        inputs = tokenizer(text, max_length=512, truncation=True, padding='longest', return_tensors="pt").to(device) 
        outputs = model.generate(inputs['input_ids'], num_beams=num_beams, max_length=max_length, num_return_sequences=num_beams)
        #output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True) 

    return output_text


def extract_from_beams(seq_list):
    total_phrases = []
    for seq in seq_list:
        phrases = seq.split(';')
        phrases = [ p.strip() for p in phrases if p.strip() != '']
        for phrase in phrases:
            if phrase not in total_phrases:
                total_phrases.append(phrase)
    return total_phrases

def remove_redundant_after_stemming(preds):
    ps = PorterStemmer()

    preds_set = []
    predicted_stem = []
    stem_preds = [ ' '.join([ps.stem(w) for w in meng17_tokenize(p)]) for p in preds]
    for p, stem_p in zip(preds, stem_preds):
        if stem_p not in predicted_stem:
            predicted_stem.append(stem_p)
            preds_set.append(p)
    return preds_set


ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

total_preds = []
total_present_preds = []
total_absent_preds = []

total_labels = []
total_present_labels = []
total_absent_labels = []


for d in tqdm(data):

  text = d['title'] + '. ' +  d['abstract']
  text = text.lower()

  extracted_phrases_seq = generate_keyphrases(model, text, tokenizer, device=device, max_length=512, num_beams=num_beams)
  extracted_phrases = extract_from_beams(extracted_phrases_seq)

  preprocessed_text = meng17_tokenize(text)
  stem_text = ' '.join([ps.stem(w) for w in preprocessed_text])
  
  extracted_phrases = [p.lower() for p in extracted_phrases if p.split()[0] not in stop_words and p.split()[-1] not in stop_words]

  present_pred = [p for p in extracted_phrases if ' '.join([ps.stem(w) for w in meng17_tokenize(p)]) in stem_text]
  absent_pred = [p for p in extracted_phrases if ' '.join([ps.stem(w) for w in meng17_tokenize(p)]) not in stem_text]
  
  labels = d['keywords'].split(";")
  labels = [ p.strip() for p in labels if p.strip() != '']

  present_label = [p for p in labels if ' '.join([ps.stem(w) for w in meng17_tokenize(p)]) in stem_text]
  absent_label = [p for p in labels if ' '.join([ps.stem(w) for w in meng17_tokenize(p)]) not in stem_text]


  total_preds.append(extracted_phrases)
  total_present_preds.append(present_pred)
  total_absent_preds.append(absent_pred)

  total_labels.append(present_label + absent_label)
  total_present_labels.append(present_label)
  total_absent_labels.append(absent_label)
  

print('total keyphrase prediction score:')
total_evaluate(total_preds, total_labels)


print('present keyphrase prediction score:')
total_evaluate(total_present_preds, total_present_labels)


print('absent keyphrase prediction score:')
total_evaluate(total_absent_preds, total_absent_labels)
