#  from pythainlp.tokenize import word_tokenize, dict_word_tokenize
import deepcut

tokens = []

with open('discard.txt', 'r') as df:
    discard = [x.decode('utf-8').strip().lower() for x in df.readlines()]
    with open('corpus/source_list.txt', 'r') as src_list:
        for src in src_list:
            with open('corpus/sources/' + src.rstrip(), 'r') as sf:
                text = ' '.join([x.decode('utf-8').strip().lower() for x in sf.readlines()])
                _tokens = [x.strip().replace(u' ', u'') for x in deepcut.tokenize(text, custom_dict='dict.txt') if x.strip().replace(u' ', u'') != u'']
                tokens += [x for x in _tokens if x not in discard]

print(' '.join(tokens).encode('utf-8'))
