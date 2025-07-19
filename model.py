import torch.nn as nn


class RNNTagger(nn.Module):
    def __init__(self, vocab_size, target_size, embedding_dim, hidden_dim):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.hidden_to_tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, _ = self.rnn(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden_to_tag(rnn_out.view(len(sentence), -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)

        return tag_scores


class NLPObject:
    def __init__(self, nlp_tokens, nlp_ents):
        self._tokens = nlp_tokens
        self._ents = nlp_ents
