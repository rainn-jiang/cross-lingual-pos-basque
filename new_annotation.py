# This script annotates the held-out Basque dev split using the better-performing system (System B),
# and reports evaluation metrics on the dev split as an independent reliability check.

from conllu import parse_incr
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

from data_set_prep import load_conllu_file, label_to_id, id_to_label, tags

tokeniser = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Load the fine-tuned System B model (adjust path to wherever your System B checkpoint was saved)
model = AutoModelForTokenClassification.from_pretrained(
    "./system_b_output/checkpoint-676",
    num_labels=len(label_to_id),
    id2label=id_to_label,
    label2id=label_to_id,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokeniser)


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
    from datasets import Dataset
    all_input_ids, all_attention_mask, all_labels = [], [], []
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


def annotate_conllu_with_predictions(input_filename, output_filename, predicted_tags_per_sentence):
    with open(input_filename, "r", encoding="utf-8") as infile:
        tokenlists = list(parse_incr(infile))

    with open(output_filename, "w", encoding="utf-8") as outfile:
        for tokenlist, predicted_tags in zip(tokenlists, predicted_tags_per_sentence):
            tag_index = 0
            for tok in tokenlist:
                if isinstance(tok["id"], int):
                    tok["upostag"] = predicted_tags[tag_index]
                    tag_index += 1
            outfile.write(tokenlist.serialize())


if __name__ == "__main__":

    basque_dev_set = load_conllu_file("eu_bdt-ud-dev.conllu")
    print("number of basque dev sentences:", len(basque_dev_set))

    basque_dev_dataset = prepare_dataset(basque_dev_set, tokeniser, label_to_id)


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
    )
    predictions_output = trainer.predict(basque_dev_dataset)

    predicted_label_ids = np.argmax(predictions_output.predictions, axis=2)

    predicted_tags_per_sentence = [
        [id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predicted_label_ids, predictions_output.label_ids)
    ]
    true_tags_per_sentence = [
        [id_to_label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predicted_label_ids, predictions_output.label_ids)
    ]


    print("\nSystem B on Basque Dev Set (Annotation Reliability Check)")
    print("Precision:", precision_score(true_tags_per_sentence, predicted_tags_per_sentence))
    print("Recall:", recall_score(true_tags_per_sentence, predicted_tags_per_sentence))
    print("F1:", f1_score(true_tags_per_sentence, predicted_tags_per_sentence))
    print("Accuracy:", accuracy_score(true_tags_per_sentence, predicted_tags_per_sentence))


    annotate_conllu_with_predictions(
        "eu_bdt-ud-dev.conllu",
        "eu_bdt-ud-dev.system_b_annotated.conllu",
        predicted_tags_per_sentence,
    )
    print("\nAnnotated corpus written to eu_bdt-ud-dev.system_b_annotated.conllu")