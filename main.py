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
from model import RNNTagger, NLPObject
import pandas as pd
import matplotlib.pyplot as plt

sentences = treebank.tagged_sents(tagset="universal")
training_data = [([w for w, t in sent], [t for w, t in sent]) for sent in sentences]

turkish_names = pd.read_csv("./datasets/turkish_names.csv")

for name in turkish_names["names"]:
    name = str(name).strip().lower()
    if name:
        training_data.append(([name], ["NOUN"]))


with open("./datasets/pos_dataset.json", "r") as fp:
    output = json.load(fp)
    languages = output["languages"]
    tag_mapping = {
        "PUNCT": ".",
        "NOUN": "NOUN",
        "VERB": "VERB",
        "ADJ": "ADJ",
        "ADV": "ADV",
        "PRON": "PRON",
        "DET": "DET",
        "ADP": "ADP",
        "NUM": "NUM",
        "CONJ": "CONJ",
        "PRT": "PRT",
        "X": "X",
    }
    all_tags = []
    for language in tqdm(languages, desc="Languages"):
        for data in tqdm(
            language["dataset"],
            desc="Dataset are merging",
            total=len(language["dataset"]),
        ):
            tokens = word_tokenize(data["sentence"])
            tokens = [token.lower() for token in tokens]
            mapped_tags = []
            for token_data in data["tokens"]:
                pos_tag = token_data["pos"]
                mapped_tag = tag_mapping.get(pos_tag, "X")
                mapped_tags.append(mapped_tag)

            if len(tokens) == len(mapped_tags):
                training_data.append((tokens, mapped_tags))
            else:
                # print(f"Mismatch: {len(tokens)} tokens vs {len(mapped_tags)} tags")
                continue

# print(f"Total POS TAG Count: {len(training_data)}")
word_to_ix = {}
tags_to_ix = {}

for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tags_to_ix:
            tags_to_ix[tag] = len(tags_to_ix)


def prepare_sequence(seq, to_ix):
    try:
        return torch.tensor([to_ix.get(w, 0) for w in seq], dtype=torch.long)
    except Exception as e:
        print(f"Error processing sequence: {seq}")
        print(f"Error: {e}")
        return torch.tensor([0], dtype=torch.long)


def pos_tag(text: str):
    model = RNNTagger(len(word_to_ix), len(tags_to_ix), 64, 64)
    model.load_state_dict(torch.load("./out/trained_pos_tagger.pth"))
    model.eval()

    with torch.no_grad():
        tokens = word_tokenize(text.lower())
        inputs = prepare_sequence(tokens, word_to_ix)
        tag_scores = model(inputs)

        predicted_tags = torch.argmax(tag_scores, dim=1)
        ix_to_tags = {v: k for k, v in tags_to_ix.items()}
        predicted_tag_names = [ix_to_tags[ix.item()] for ix in predicted_tags]

        tokens_and_tags = []
        for i in range(len(tokens)):
            tokens_and_tags.append((tokens[i], predicted_tag_names[i]))

        object = NLPObject(tokens, tokens_and_tags)

        return object


def main():
    model = RNNTagger(len(word_to_ix), len(tags_to_ix), 64, 64)

    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    plt.title = "Loss Data"
    plt_x = []  # Loss per epoch
    plt_y = []  # Epoch
    for epoch in tqdm(range(30), desc="Training..."):  # 30 epoch
        total_loss = 0
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tags_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        plt_x.append(total_loss)
        plt_y.append(epoch)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    plt.scatter(plt_x, plt_y)
    plt.show()
    os.makedirs("./out", exist_ok=True)
    torch.save(model.state_dict(), "./out/trained_pos_tagger.pth")


if __name__ == "__main__":
    main()
