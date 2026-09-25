#This script is for System B construction.

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from  data_set_prep import load_conllu_file
from data_set_prep import label_to_id, id_to_label, tags
from datasets import Dataset
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

tokeniser = AutoTokenizer.from_pretrained("xlm-roberta-base")

model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels = len(label_to_id),
    id2label = id_to_label,
    label2id = label_to_id,
)

data_collator = DataCollatorForTokenClassification(tokenizer = tokeniser)

training_args = TrainingArguments(
    output_dir = "./system_b_output",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    num_train_epochs = 3,
    weight_decay = 0.01,
)

def tokenise_and_align_labels(words, tags, tokeniser, label_to_id):
    tokenised = tokeniser(words, is_split_into_words=True, truncation=True)

    word_ids = tokenised.word_ids()

    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label_to_id[tags[word_idx]])
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenised["labels"] = labels
    return tokenised

def prepare_dataset(sentence_list, tokeniser, label_to_id):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for words, tags in sentence_list:
        result = tokenise_and_align_labels(words, tags, tokeniser, label_to_id)
        all_input_ids.append(result["input_ids"])
        all_attention_mask.append(result["attention_mask"])
        all_labels.append(result["labels"])

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

if __name__ == "__main__":
    basque_train_set = load_conllu_file("eu_bdt-ud-train.conllu")
    basque_test_set = load_conllu_file("eu_bdt-ud-test.conllu")

    basque_train_dataset = prepare_dataset(basque_train_set, tokeniser, label_to_id)
    print("number of basque training:", len(basque_train_dataset))


    basque_test_dataset = prepare_dataset(basque_test_set, tokeniser, label_to_id)
    print("number of basque testing:", len(basque_test_dataset))

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = basque_train_dataset,
        eval_dataset = basque_test_dataset,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()


    results = trainer.evaluate()
    print("System B on Basque:")
    print(results)