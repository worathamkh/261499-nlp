# -*- coding: utf-8 -*-

import argparse
import numpy as np
#  from pythainlp.tokenize import word_tokenize
import deepcut
from sklearn.preprocessing import OneHotEncoder

def is_numeric(s):
    return any(c.isdigit() for c in s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.decode('utf-8').rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.decode('utf-8').rstrip().split(' ')
            vectors[vals[0]] = map(float, vals[1:])

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.iteritems():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    evaluate_vectors(W_norm, vocab, ivocab)

def evaluate_vectors(W, vocab, ivocab):
    """Evaluate the trained word vectors on a variety of tasks"""

    #  discards = [u' ', u'เท่าไหร่', u'เท่า', u'อะไร', u'ไหน', u'ไหร่', u'กี่', u'ที่ไหน', u'ที่ใด', u'ใด', u'แห่งใด', u'ใคร']

    questions_file = 'question_list.txt'
    ans_file = 'expected_ans.txt'
    prefix = './eval/question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    vocab_size = W.shape[0]
    vector_dim = W.shape[1]

    onehot_encoder = OneHotEncoder(sparse=False, n_values=vocab_size)

    with open('discard.txt', 'r') as df, \
        open('numeric.txt', 'r') as nf, \
        open('person.txt', 'r') as pf, \
        open('location.txt', 'r') as lf, \
        open('%s/%s' % (prefix, questions_file), 'r') as qf, \
        open('%s/%s' % (prefix, ans_file), 'r') as af:
        discard = [x.decode('utf-8').strip().lower() for x in df.readlines()]
        discard += [u' ']
        numeric = set([x.decode('utf-8').strip().lower() for x in nf.readlines()])
        person = set([x.decode('utf-8').strip().lower() for x in pf.readlines()])
        location = set([x.decode('utf-8').strip().lower() for x in lf.readlines()])
        questions = [x.split('::')[1].strip().decode('utf-8') for x in qf.readlines()]
        ans = [x.split('::')[1].strip().decode('utf-8') for x in af.readlines()]
        for i in xrange(len(questions)):
            tokens = deepcut.tokenize(questions[i], custom_dict='dict.txt')
            context = [word.strip().replace(u' ', u'') for word in tokens if word not in discard]
            context_idx = np.array([vocab[word] for word in context if word in vocab])
            context_1hot = np.sum(onehot_encoder.fit_transform(context_idx.reshape(len(context_idx), 1)), axis=0)

            # adjust weight by question type: person, location
            if not person.isdisjoint(tokens):
                pass
            elif not location.isdisjoint(tokens):
                pass

            #  print(context_1hot)
            context_vec = np.dot(context_1hot.T, W).T # / len(context_idx)
            #  print(context_vec)
            ans_1hot = np.dot(W, context_vec.T)
            #  print(ans_1hot)
            #  predicted_ans = ivocab[np.argmax(ans_1hot)]
            predictions = [ivocab[idx] for idx in np.argsort(ans_1hot)[-len(ans_1hot):] if idx not in context_idx]
            predictions.reverse()

            # reduce predictions if question is numeric
            if not numeric.isdisjoint(tokens):
                predictions = [x for x in predictions if is_numeric(x)]

            print('question: %s' % questions[i].encode('utf-8'))
            print('context: %s' % '|'.join(context).encode('utf-8'))
            print('predictions: %s' % ', '.join(predictions[0:10]).encode('utf-8'))
            print('expected: %s' % ans[i].encode('utf-8'))
            print('-----')

    #  for i in xrange(len(filenames)):
    #      with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
    #          full_data = [line.rstrip().split(' ') for line in f]
    #          full_count += len(full_data)
    #          data = [x for x in full_data if all(word in vocab for word in x)]
    #
    #      indices = np.array([[vocab[word] for word in row] for row in data])
    #      ind1, ind2, ind3, ind4 = indices.T
    #
    #      predictions = np.zeros((len(indices),))
    #      num_iter = int(np.ceil(len(indices) / float(split_size)))
    #      for j in xrange(num_iter):
    #          subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))
    #
    #          pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
    #              +  W[ind3[subset], :])
    #          #cosine similarity if input W has been normalized
    #          dist = np.dot(W, pred_vec.T)
    #
    #          for k in xrange(len(subset)):
    #              dist[ind1[subset[k]], k] = -np.Inf
    #              dist[ind2[subset[k]], k] = -np.Inf
    #              dist[ind3[subset[k]], k] = -np.Inf
    #
    #          # predicted word index
    #          predictions[subset] = np.argmax(dist, 0).flatten()
    #
    #      val = (ind4 == predictions) # correct predictions
    #      count_tot = count_tot + len(ind1)
    #      correct_tot = correct_tot + sum(val)
    #      if i < 5:
    #          count_sem = count_sem + len(ind1)
    #          correct_sem = correct_sem + sum(val)
    #      else:
    #          count_syn = count_syn + len(ind1)
    #          correct_syn = correct_syn + sum(val)
    #
    #      print("%s:" % filenames[i])
    #      print('ACCURACY TOP1: %.2f%% (%d/%d)' %
    #          (np.mean(val) * 100, np.sum(val), len(val)))
    #
    #  print('Questions seen/total: %.2f%% (%d/%d)' %
    #      (100 * count_tot / float(full_count), count_tot, full_count))
    #  print('Semantic accuracy: %.2f%%  (%i/%i)' %
    #      (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    #  print('Syntactic accuracy: %.2f%%  (%i/%i)' %
    #      (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    #  print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


if __name__ == "__main__":
    main()
