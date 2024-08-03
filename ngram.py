input = [
    "I love natural language processing",
    "natural language processing is amazing",
    "I love programming"
]

def generate_vocab(input_sentences):
    tokens = {
        "<s>": 0,
        "</s>": 1,
    }
    words = {
        0: "<s>",
        1: "</s>",
    }

    token_index = 2
    for sentence in input_sentences:
        for word in sentence.lower().split(" "):
            if word not in words:
                tokens[word] = token_index
                words[token_index] = word
                token_index += 1
    return (tokens, words)

def tokenize(input_sentences, tokens):
    full_text = []
    for sentence in input_sentences:
        full_text.append(tokens["<s>"])
        for word in sentence.lower().split(" "):
            full_text.append(tokens[word])
        full_text.append(tokens["</s>"])
    return full_text

def detokenize_bigrams(bigrams, word_vocab):
    out = {}
    for key, value in bigrams.items():
        f, s = key
        f = word_vocab[f]
        s = word_vocab[s]
        out[(f, s)] = value
    return out

def get_all_bigrams(vocab_tokens):
    all = {}
    for first in vocab_tokens.values():
        for second in vocab_tokens.values():
            all[(first, second)] = 0
    return all

def smooth_bigram_counts(bigram_counts, total):
    for bigram in bigram_counts.keys():
        bigram_counts[bigram] += 1
    return total + len(bigram_counts.keys())

def count_bigrams(input, vocab):
    bigram_counts = get_all_bigrams(vocab)
    total = 0
    for i in range(len(input)-1):
        current_bigram = (input[i], input[i+1])
        bigram_counts[current_bigram] += 1
        total += 1
    return (bigram_counts, total)

def prob_from_bigram_count(bigram_counts, total):
    bigrams = bigram_counts.copy()
    for bigram, count in bigram_counts.items():
        bigrams[bigram] = bigram_counts[bigram] / total
    return bigrams
    
def get_bigrams(input):
    token_vocab, word_vocab = generate_vocab(input)
    input_tokens = tokenize(input, token_vocab)
    bigram_counts, total = count_bigrams(input_tokens, token_vocab)
    total = smooth_bigram_counts(bigram_counts, total)
    bigram_probability = prob_from_bigram_count(bigram_counts, total)
    out = detokenize_bigrams(bigram_probability, word_vocab)
    return out

a = get_bigrams(input)

for key, value in a.items():
    print(f"{value}\t{key}")
