import tensorflow as tf
import sys
import numpy as np
from attention_wrapper import *
from tensorflow.python.ops import array_ops
import logging
import time
from progbar_class import *


logging.basicConfig(stream = sys.stdout, level=logging.INFO)

class Config:
    num_epochs = 30
    batch_size = 32
    train_embeddings=0
    max_gradient_norm=-1
    hidden_state_size=150
    embedding_size=50
    data_dir="data/squad"
    vocab_path="data/squad/vocab.dat"
    embed_path="data/squad/glove.trimmed.50.npz"
    dropout_val=1.0
    train_dir="models_lstm_basic"
    use_match=0
    

    def get_paths(mode):
        question = "data/squad/%s.ids.questions" %mode
        context = "data/squad/%s.ids.contexts" %mode
        answer = "data/squad/%s.spans" %mode

        return question, context, answer 

    question_train, context_train, answer_train = get_paths("train")
    question_dev ,context_dev ,answer_dev = get_paths("val")



class squad_dataset(object):
    def __init__(self, question_file, context_file, answer_file):
        """
        Args:
            filename: path to the files
        """
        self.question_file = question_file
        self.context_file = context_file
        self.answer_file = answer_file

        self.length = None

    def iter_file(self, filename):
        with open(filename) as f:
            for line in f:
                line = line.strip().split(" ")
                line = map(lambda tok: int(tok), line)
                yield line


    def __iter__(self):
        niter = 0

        question_file_iter = self.iter_file(self.question_file)
        answer_file_iter = self.iter_file(self.answer_file)
        context_file_iter = self.iter_file(self.context_file)

        for question, context, answer in zip(question_file_iter, context_file_iter, answer_file_iter):
            yield list(question),list(context), list(answer)



    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        nmatrix of embeddings (np array)
    """
    return np.load(filename)["glove"]

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip(b'\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

    
config = Config()
train = squad_dataset(config.question_train, config.context_train, config.answer_train)
dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)



embed_path = config.embed_path
vocab_path = config.vocab_path
vocab, rev_vocab = initialize_vocab(vocab_path)
embeddings = get_trimmed_glove_vectors(embed_path)


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return np.array(sequence_padded), np.array(sequence_length)


def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_length = max([len(list(x)) for x in sequences])
    sequence_padded, sequence_length = _pad_sequences(sequences, 
                                            pad_tok, max_length)

    return sequence_padded, sequence_length    


question_ids = tf.placeholder(tf.int32, shape = [None, None], name = "question_ids")
passage_ids = tf.placeholder(tf.int32, shape = [None, None], name = "passage_ids")

question_lengths = tf.placeholder(tf.int32, shape=[None], name="question_lengths")
passage_lengths = tf.placeholder(tf.int32, shape = [None], name = "passage_lengths")

labels = tf.placeholder(tf.int32, shape = [None, 2], name = "gold_labels")
dropout = tf.placeholder(tf.float32, shape=[], name = "dropout")




def get_feed_dict(questions, contexts, answers, dropout_val):
    """
    -arg questions: A list of list of ids representing the question sentence
    -arg contexts: A list of list of ids representing the context paragraph
    -arg dropout_val: A float representing the keep probability for dropout 

    :return: dict {placeholders: value}
    """

    padded_questions, question_length = pad_sequences(questions, 0)
    padded_contexts, passage_length = pad_sequences(contexts, 0)


    feed = {
        question_ids : padded_questions,
        passage_ids : padded_contexts,
        question_lengths : question_length,
        passage_lengths : passage_length,
        labels : answers,
        dropout : dropout_val
    }

    return feed



with tf.variable_scope("vocab_embeddings"):
    _word_embeddings = tf.Variable(embeddings, name="_word_embeddings", dtype=tf.float32, trainable= config.train_embeddings)
    question_emb = tf.nn.embedding_lookup(_word_embeddings, question_ids, name = "question") # (-1, Q, D)
    passage_emb = tf.nn.embedding_lookup(_word_embeddings, passage_ids, name = "passage") # (-1, P, D)
    # Apply dropout
    question = tf.nn.dropout(question_emb, config.dropout_val)
    passage  = tf.nn.dropout(passage_emb, config.dropout_val)

hidden_size=150
def encode(inputs, masks, encoder_state_input = None):
    """
    :param inputs: vector representations of question and passage (a tuple) 
    :param masks: masking sequences for both question and passage (a tuple)

    :param encoder_state_input: (Optional) pass this as initial hidden state
                                to tf.nn.dynamic_rnn to build conditional representations
    :return: an encoded representation of the question and passage.
    """


    question, passage = inputs
    masks_question, masks_passage = masks    


    # read passage conditioned upon the question
    with tf.variable_scope("encoded_question"):
        lstm_cell_fw_question = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple = True)
        lstm_cell_bw_question = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple = True)
        encoded_question, (q_rep, _) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw_question, lstm_cell_bw_question, question, masks_question, dtype=tf.float32) # (-1, Q, H)

    with tf.variable_scope("encoded_passage"):
        lstm_cell_fw_passage  = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple = True)
        lstm_cell_bw_passage  = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple = True)
        encoded_passage, (p_rep, _) =   tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw_passage, lstm_cell_bw_passage, passage, masks_passage, dtype=tf.float32) # (-1, P, H)

    # Merging both the outputs of the bi-lstm models
    encoded_question = tf.concat(axis = 2, values = encoded_question)
    encoded_passage = tf.concat(axis = 2, values = encoded_passage)

    # outputs beyond sequence lengths are masked with 0s
    return encoded_question, encoded_passage , q_rep, p_rep    



def _reverse(input_, seq_lengths, seq_dim, batch_dim):
  if seq_lengths is not None:
    return array_ops.reverse_sequence(
        input=input_, seq_lengths=seq_lengths,
        seq_dim=seq_dim, batch_dim=batch_dim)
  else:
    return array_ops.reverse(input_, axis=[seq_dim])

# Match LSTM
def run_match_lstm(encoded_rep, masks):
    encoded_question, encoded_passage = encoded_rep
    masks_question, masks_passage = masks
    
    match_lstm_cell_attention_fn = lambda curr_input, state : tf.concat([curr_input, state], axis = -1)
    query_depth = encoded_question.get_shape()[-1]
    
    with tf.variable_scope("match_lstm_attender"):
        attention_mechanism_match_lstm = BahdanauAttention(query_depth, encoded_question, memory_sequence_length = masks_question)
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple = True)
        lstm_attender  = AttentionWrapper(cell, attention_mechanism_match_lstm, output_attention = False, attention_input_fn = match_lstm_cell_attention_fn)
    
        # we don't mask the passage because masking the memories will be handled by the pointerNet
        reverse_encoded_passage = _reverse(encoded_passage, masks_passage, 1, 0)
    
        output_attender_fw, _ = tf.nn.dynamic_rnn(lstm_attender, encoded_passage, dtype=tf.float32, scope ="rnn")    
        output_attender_bw, _ = tf.nn.dynamic_rnn(lstm_attender, reverse_encoded_passage, dtype=tf.float32, scope = "rnn")
    
        output_attender_bw = _reverse(output_attender_bw, masks_passage, 1, 0)
    
    output_attender = tf.concat([output_attender_fw, output_attender_bw], axis = -1) # (-1, P, 2*H)
    return output_attender
    

# Answer Pointer
def run_answer_ptr(output_attender, masks, labels):
    #batch_size = tf.shape(output_attender)[0]
    masks_question, masks_passage = masks
    labels = tf.unstack(labels, axis=1) 
    
    answer_ptr_cell_input_fn = lambda curr_input, context : context # independent of question
    query_depth_answer_ptr = output_attender.get_shape()[-1]
    
    with tf.variable_scope("answer_ptr_attender"):
        attention_mechanism_answer_ptr = BahdanauAttention(query_depth_answer_ptr , output_attender, memory_sequence_length = masks_passage)
    
        # output attention is true because we want to output the attention values
        cell_answer_ptr = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple = True )
        answer_ptr_attender = AttentionWrapper(cell_answer_ptr, attention_mechanism_answer_ptr, cell_input_fn = answer_ptr_cell_input_fn)
        logits, _ = tf.nn.static_rnn(answer_ptr_attender, labels, dtype = tf.float32)
    
    return logits


# decoder
def decode(encoded_rep, q_rep, masks, labels):
    """
    takes in a knowledge representation
    and output a probability estimation over
    all paragraph tokens on which token should be
    the start of the answer span, and which should be
    the end of the answer span.

    :param knowledge_rep: it is a representation of the paragraph and question,
                          decided by how you choose to implement the encoder
    :return:
    """
    # Run match-LSTM + Ans-Ptr
    output_attender = run_match_lstm(encoded_rep, masks)
    logits = run_answer_ptr(output_attender, masks, labels)
    
    return logits

# setup_system
encoded_question, encoded_passage, q_rep, p_rep = encode([question,passage], [question_lengths,passage_lengths],encoder_state_input = None)
encoded_rep = encoded_question, encoded_passage
masks = question_lengths,passage_lengths
logits = decode(encoded_rep, q_rep, masks, labels)

# setup_loss
losses= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[0], labels=labels[:,0])
losses+= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[1], labels=labels[:,1])
loss = tf.reduce_mean(losses)

max_gradient_norm = -1

# setup_train_op
with tf.variable_scope("train_step"):
    adam_optimizer = tf.train.AdamOptimizer()
    grads, varss = zip(*adam_optimizer.compute_gradients(loss))

    clip_val = max_gradient_norm
    
    # if -1 then do not perform gradient clipping
    if clip_val != -1:
        clipped_grads, _ = tf.clip_by_global_norm(grads, max_gradient_norm)
        global_grad = tf.global_norm(clipped_grads)
        gradients = zip(clipped_grads, varss)
    else:
        global_grad = tf.global_norm(grads)
        gradients = zip(grads, varss)

    train_op = adam_optimizer.apply_gradients(gradients)
    
init = tf.global_variables_initializer()
        

logger = logging.getLogger("QASystemLogger")


saver = tf.train.Saver()

def initialize_model(session, train_dir):
    """
    param: session managed from train.py
    param: train_dir : the directory in which models are saved
    """
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(init)
        logger.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

        
def test(session, valid):
    """
    valid: a list containing q, c and a.
    :return: loss on the valid dataset and the logit values
    """
    q, c, a = valid

    # at test time we do not perform dropout.
    input_feed =  get_feed_dict(q, c, a, 1.0)

    output_feed = [logits]
    outputs = session.run(output_feed, input_feed)

    return outputs[0][0], outputs[0][1]


def answer(session, dataset):    
    '''
    Get the answers for dataset. Independent of how data iteration is implemented
    '''
    yp, yp2 = test(session, dataset)
    a_s, a_e = [], []
    for i in range(yp.shape[0]):
        _a_s, _a_e = func(yp[i], yp2[i])
        a_s.append(_a_s)
        a_e.append(_a_e)
    
    return (np.array(a_s), np.array(a_e))


# -- Boundary Model with a max span restriction of 15
def func(y1, y2):
    max_ans = -999999
    a_s, a_e= 0,0
    num_classes = len(y1)
    for i in range(num_classes):
        for j in range(15):
            if i+j >= num_classes:
                break

            curr_a_s = y1[i];
            curr_a_e = y2[i+j]
            if (curr_a_e+curr_a_s) > max_ans:
                max_ans = curr_a_e + curr_a_s
                a_s = i
                a_e = i+j

    return (a_s, a_e)
    

def evaluate_model(session, dataset):
    q, c, a = zip(*[[_q, _c, _a] for (_q, _c, _a) in dataset])

    sample = len(dataset)

    a_s, a_o = answer(session, [q, c, a])
    
    answers = np.hstack([a_s.reshape([sample, -1]), a_o.reshape([sample,-1])])
    gold_answers = np.array([a for (_,_, a) in dataset])
    
    em_score = 0
    em_1 = 0
    em_2 = 0
    for i in range(sample):
        gold_s, gold_e = gold_answers[i]
        s, e = answers[i]
        if (s==gold_s): em_1 += 1.0
        if (e==gold_e): em_2 += 1.0
        if (s == gold_s and e == gold_e):
            em_score += 1.0
    
    em_1 /= float(len(answers))
    em_2 /= float(len(answers))
    logger.info("\nExact match on 1st token: %5.4f | Exact match on 2nd token: %5.4f\n" %(em_1, em_2))
    
    em_score /= float(len(answers))
    
    return em_score        
    


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (question, context, answer) tuples
        minibatch_size: (int)
    Returns: 
        list of tuples
    """
    question_batch, context_batch, answer_batch = [], [], []

    for (q, c, a) in data:
        if len(question_batch) == minibatch_size:
            yield question_batch, context_batch, answer_batch
            question_batch, context_batch, answer_batch = [], [], []
        
        question_batch.append(q)
        context_batch.append(c)
        answer_batch.append(a)

    if len(question_batch) != 0:
        yield question_batch, context_batch, answer_batch


def run_epoch(session, train):
    """
    Perform one complete pass over the training data and evaluate on dev
    """

    nbatches = (len(train) + config.batch_size - 1) / config.batch_size
    prog = Progbar(target=nbatches)

    for i, (q_batch, c_batch, a_batch) in enumerate(minibatches(train, config.batch_size)):

        # at training time, dropout needs to be on.
        input_feed = get_feed_dict(q_batch, c_batch, a_batch, config.dropout_val)

        _, train_loss = session.run([train_op, loss], feed_dict=input_feed)
        prog.update(i + 1, [("train loss", train_loss)])
        
                        

def _train(session, dataset, train_dir):
    """
    Implement main training loop

    :param session: it should be passed in from train.py
    :param dataset: a list containing the training and dev data
    :param train_dir: path to the directory where you should save the model checkpoint
    
    :return:
    """    
    if not tf.gfile.Exists(train_dir):
        tf.gfile.MkDir(train_dir)

    train, dev = dataset
    
    em = evaluate_model(session, dev)
    logger.info("\n#-----------Initial Exact match on dev set: %5.4f ---------------#\n" %em)

    best_em = 0

    for epoch in range(config.num_epochs):
        logger.info("\n*********************EPOCH: %d*********************\n" %(epoch+1))
        run_epoch(session, train)
        em = evaluate_model(session, dev)
        logger.info("\n#-----------Exact match on dev set: %5.4f #-----------\n" %em)
        #======== Save model if it is the best so far ========
        if (em > best_em):
            saver.save(session, "%s/best_model.chk" %train_dir)
            best_em = em
            
            
            
with tf.Session() as sess:
    # ====== Load a pretrained model if it exists or create a new one if no pretrained available ======
    initialize_model(sess, config.train_dir)
    _train(sess, [train, dev], config.train_dir)            
            