from __future__ import print_function
import argparse
import json
import linecache
import nltk
import numpy as np
import os
import sys
from tqdm import tqdm
import random
import linecache


from collections import Counter
from six.moves.urllib.request import urlretrieve


def data_from_json(filename):
    with open(filename,'rb') as data_file:
        data = json.load(data_file)
    return data


def list_topics(data):
    list_topics = [data['data'][idx]['title'] for idx in range(0,len(data['data']))]
    return list_topics


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return tokens


def token_idx_map(context, context_tokens):
    acc = ''
    current_token_idx = 0
    token_map = dict()

    for char_idx, char in enumerate(context):
        if char != ' ':
            acc += char
            context_token = context_tokens[current_token_idx]
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    return token_map

def read_write_dataset(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    skipped = 0

    with open(os.path.join(prefix, tier +'.context'), 'w') as context_file,  \
         open(os.path.join(prefix, tier +'.question'), 'w') as question_file,\
         open(os.path.join(prefix, tier +'.answer'), 'w') as text_file, \
         open(os.path.join(prefix, tier +'.span'), 'w') as span_file:

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens = tokenize(context)
                answer_map = token_idx_map(context, context_tokens)

                qas = article_paragraphs[pid]['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    question_tokens = tokenize(question)
                    answers = qas[qid]['answers']
                    qn += 1

                    num_answers = list(range(1))

                    for ans_id in num_answers:
                        # it contains answer_start, text
                        text = qas[qid]['answers'][ans_id]['text']

                        text_tokens = tokenize(text)

                        answer_start = qas[qid]['answers'][ans_id]['answer_start']

                        answer_end = answer_start + len(text)

                        last_word_answer = len(text_tokens[-1]) # add one to get the first char

                        try:
                            a_start_idx = answer_map[answer_start][1]

                            a_end_idx = answer_map[answer_end - last_word_answer][1]

                            # remove length restraint since we deal with it later
                            context_file.write(' '.join(context_tokens) + '\n')
                            question_file.write(' '.join(question_tokens) + '\n')
                            text_file.write(' '.join(text_tokens) + '\n')
                            span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')

                        except Exception as e:
                            skipped += 1

                        an += 1

    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return qn,an



def save_files(prefix, tier, indices):
  with open(os.path.join(prefix, tier + '.contexts'), 'w') as context_file,  \
      open(os.path.join(prefix, tier + '.questions'), 'w') as question_file,\
      open(os.path.join(prefix, tier + '.answers'), 'w') as text_file, \
      open(os.path.join(prefix, tier + '.spans'), 'w') as span_file, \
      open(os.path.join(prefix, 'train.context'), 'rb') as context, \
      open(os.path.join(prefix, 'train.question'),'rb') as question, \
      open(os.path.join(prefix, 'train.answer'),'rb') as answer, \
      open(os.path.join(prefix, 'train.span'),'rb') as span:
      
      line1=context.readlines()
      line2=question.readlines()
      line3=answer.readlines()
      line4=span.readlines()
                
      for i in indices: 
          context_file.write(str(line1[i])+'\n')
          question_file.write(str(line2[i])+'\n')
          text_file.write(str(line3[i])+'\n')
          span_file.write(str(line4[i])+'\n')


def split_tier(prefix, train_percentage, shuffle=False):

    context_filename = os.path.join(data_prefix, 'train' + '.context')

    with open(context_filename,'rb') as current_file:
        num_lines = sum(1 for line in current_file)

    # Get indices and split into two files
    indices_dev = list(range(num_lines)[int(num_lines * train_percentage)::])
    if shuffle:
        np.random.shuffle(indices_dev)
        print("Shuffling...")
    save_files(prefix, 'val', indices_dev)
    indices_train = list(range(num_lines)[:int(num_lines * train_percentage)])
    if shuffle:
        np.random.shuffle(indices_train)
    save_files(prefix, 'train', indices_train)
        

if __name__ == '__main__':

    download_prefix = os.path.join("download", "squad")
    data_prefix = os.path.join("data", "squad")

    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"

    train_data = data_from_json(os.path.join(download_prefix, train_filename))

    train_num_questions, train_num_answers = read_write_dataset(train_data, 'train', data_prefix)

    print("Splitting the dataset into train and validation")
    split_tier(data_prefix, 0.95, shuffle=True)

    print("Processed {} questions and {} answers in train".format(train_num_questions, train_num_answers))
    
    
    
    os.rename('data\\squad\\train.spans','data\\squad\\ntrain.spans')   
    with open('data\\squad\\ntrain.spans','r') as file:
        l=file.readlines()
        
    with open('data\\squad\\train.spans','w') as file:
        for i in range(len(l)):
            line=l[i]
            line=line[2:-6]
            file.write(line+'\n')
    
    os.rename('data\\squad\\val.spans','data\\squad\\nval.spans')            
    with open('data\\squad\\nval.spans','r') as file:
        l=file.readlines()
    with open('data\\squad\\val.spans','w') as file:
        for i in range(len(l)):
            line=l[i]
            line=line[2:-6]
            file.write(line+'\n')      
