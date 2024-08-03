from torch import nn, optim
import torch
from gensim.models import Word2Vec

NGRAM_SIZE = 2
EMBEDDING_SIZE = 100
TRAINING_ROUNDS = 500
LAYER_SIZE = 1000
LEARNING_RATE = 0.001


def train_word2vec(input, vector_dim):
    word2vec = Word2Vec(sentences=input, vector_size=vector_dim, window=2, min_count=1)
    return word2vec


def get_vocab(input):
    vocab = set([word for s in input for word in s.split(" ")] + ["<s>", "</s>"])
    vocab_index = {word: index for index, word in enumerate(vocab)}
    return vocab_index


def get_ngrams(input):
    ngrams = []
    for s in input_sentences:
        sentence = ["<s>"]
        sentence += s.split(" ")
        sentence.append("</s>")
        for center_index in range(NGRAM_SIZE, len(sentence)):
            context = []
            for context_index in range(NGRAM_SIZE):
                context.append(sentence[center_index - context_index - 1])
            ngrams.append((context, sentence[center_index]))
    return ngrams


class LangModel(nn.Module):
    def __init__(self, vocab_size, embedding, context_size):
        super(LangModel, self).__init__()
        self.embeddings = embedding
        self.hidden_layer = nn.Linear(context_size * EMBEDDING_SIZE, LAYER_SIZE)
        self.hidden_activiation = torch.nn.ReLU()
        self.output_layer = nn.Linear(LAYER_SIZE, vocab_size)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).view((1, -1))
        out = self.hidden_layer(embeddings)
        out = self.hidden_activiation(out)
        out = self.output_layer(out)
        log_probability = nn.functional.log_softmax(out, dim=1)
        return log_probability


def train_model(model, vocab_index, ngrams):
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(TRAINING_ROUNDS):
        total_loss = 0
        for context, center in ngrams:
            context_index = torch.tensor([vocab_index[w] for w in context])
            model.zero_grad()
            log_probs = model(context_index)
            loss = loss_function(log_probs, torch.tensor([vocab_index[center]]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


def vocab_word(target_index):
    for word, index in vocab_index.items():
        if target_index == index:
            return word


# context should be exactly CONTEXT_SIZE
def predict_next_word(model, context):
    log_probabilities = model(torch.tensor([vocab_index[cword] for cword in context]))
    best = torch.argmax(log_probabilities)
    word = vocab_word(best)
    return word


input_sentences = [
    "I love natural language processing",
    "natural language processing is amazing",
    "I love programming",
]

word2vec = train_word2vec(input_sentences, EMBEDDING_SIZE)
weights = torch.FloatTensor(word2vec.wv.vectors)
embedding = torch.nn.Embedding.from_pretrained(weights)
vocab_index = get_vocab(input_sentences)
ngrams = get_ngrams(input_sentences)
model = LangModel(len(vocab_index), embedding, NGRAM_SIZE)

try:
    model.load_state_dict(torch.load("model.pt"))
except:
    train_model(model, vocab_index, ngrams)
    torch.save(model.state_dict(), "model.pt")

model.eval()

print(predict_next_word(model, ("natural", "language")))
