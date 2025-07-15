import nltk
nltk.download("treebank")
nltk.download("universal_tagset")
from nltk.corpus import treebank
import torch
import torch.nn as nn
from nltk import word_tokenize
import json
from tqdm import tqdm
import os

sentences = treebank.tagged_sents(tagset="universal")
training_data = [([w for w, t in sent], [t for w, t in sent]) for sent in sentences]

with open("./datasets/pos_dataset.json", "r") as fp:
    output = json.load(fp)
    languages = output["languages"]
    tag_mapping = {
        'PUNCT': '.',
        'NOUN': 'NOUN',
        'VERB': 'VERB',
        'ADJ': 'ADJ',
        'ADV': 'ADV',
        'PRON': 'PRON',
        'DET': 'DET',
        'ADP': 'ADP',
        'NUM': 'NUM',
        'CONJ': 'CONJ',
        'PRT': 'PRT',
        'X': 'X'
    }

    for language in tqdm(languages, desc="Languages"):
        for data in tqdm(language["dataset"], desc="Dataset are merging", total=len(language["dataset"])):
            tokens = word_tokenize(data["sentence"])
            mapped_tags = []
            for token_data in data["tokens"]:
                pos_tag = token_data["pos"]
                mapped_tag = tag_mapping.get(pos_tag, 'X')
                mapped_tags.append(mapped_tag)
            
            if len(tokens) == len(mapped_tags):
                training_data.append((tokens, mapped_tags))
            else:
                print(f"Mismatch: {len(tokens)} tokens vs {len(mapped_tags)} tags")
                continue

    print(f"Toplam POST TAG Sayısı: {len(training_data)}")

word_to_ix = {}
tags_to_ix = {}

for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tags_to_ix:
            tags_to_ix[tag] = len(tags_to_ix)

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

def prepare_sequence(seq, to_ix):
    try:
        return torch.tensor([to_ix.get(w, 0) for w in seq], dtype=torch.long)
    except Exception as e:
        print(f"Error processing sequence: {seq}")
        print(f"Error: {e}")
        return torch.tensor([0], dtype=torch.long)


def main():
    model = RNNTagger(len(word_to_ix), len(tags_to_ix), 6, 6)

    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(10):  # 10 epochs
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tags_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
    
    os.makedirs("./out", exist_ok=True)
    torch.save(model, "./out/trained_post_tagger.pth")

    # This was for the development steps. You won't need this anymore
    """while False: 
        text = input("Text: ")
        with torch.no_grad():
            tokens = word_tokenize(text)
            inputs = prepare_sequence(tokens, word_to_ix)
            tag_scores = model(inputs)

            predicted_tags = torch.argmax(tag_scores, dim=1)
            ix_to_tags = {v: k for k, v in tags_to_ix.items()}
            predicted_tag_names = [ix_to_tags[ix.item()] for ix in predicted_tags]

            for word, tag in zip(tokens, predicted_tag_names):
                print(f"{word} -> {tag}")"""

if __name__ == "__main__":
    main()
