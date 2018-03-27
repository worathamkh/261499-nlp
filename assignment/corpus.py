from pythainlp.tokenize import word_tokenize, dict_word_tokenize

tokens = []

with open('corpus/source_list.txt', 'r') as src_list:
    for src in src_list:
        with open('corpus/sources/' + src.rstrip(), 'r') as f:
            text = ' '.join([x.decode('utf-8').lower() for x in f.readlines()])
            tokens += [x.strip().replace(' ', '') for x in word_tokenize(text, engine='deepcut') if x.strip().replace(' ', '') != '']
            #  tokens += [x.replace(' ', '') for x in dict_word_tokenize(text, "dict.txt") if x.strip() != '']

print(' '.join(tokens).encode('utf-8'))
