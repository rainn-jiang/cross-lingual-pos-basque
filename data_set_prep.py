#This script is for preliminary data cleaning and preparation.

from conllu import parse_incr

def load_conllu_file(filename):
    sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words = []
            tags = []
            for tok in tokenlist:
                if isinstance(tok["id"], int):
                    words.append(tok["form"])
                    tags.append(tok["upostag"])
            sentences.append((words, tags))
    return sentences

english_train_set = load_conllu_file("en_ewt-ud-train.conllu")
basque_train_set = load_conllu_file("eu_bdt-ud-train.conllu")
basque_test_set = load_conllu_file("eu_bdt-ud-test.conllu")

tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
label_to_id = {tag: idx for idx, tag in enumerate(tags)}
id_to_label = {idx: tag for tag, idx in label_to_id.items()}
