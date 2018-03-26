from pythainlp.tokenize import word_tokenize

tokens = []

with open('corpus/source_list.txt', 'r') as src_list:
    for src in src_list:
        with open('corpus/sources/' + src.rstrip(), 'r') as f:
            text = ' '.join([x.decode('utf-8') for x in f.readlines()])
            tokens += [x.strip() for x in word_tokenize(text) if x.strip() != '']

print(' '.join(tokens).encode('utf-8'))
