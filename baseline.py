from collections import Counter, defaultdict
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from data_set_prep import load_conllu_file

def train_frequency_baseline(train_set):

    word_tag_counts = defaultdict(Counter)
    overall_tag_counts = Counter()

    for words, tags in train_set:
        for word, tag in zip(words, tags):
            word_tag_counts[word][tag] += 1
            overall_tag_counts[tag] += 1


    word_to_most_common_tag = {
        word: tag_counter.most_common(1)[0][0]
        for word, tag_counter in word_tag_counts.items()
    }


    fallback_tag = overall_tag_counts.most_common(1)[0][0]

    return word_to_most_common_tag, fallback_tag


def predict_with_frequency_baseline(test_set, word_to_most_common_tag, fallback_tag):
    predicted_tags_per_sentence = []
    gold_tags_per_sentence = []

    for words, tags in test_set:
        predicted_tags = [
            word_to_most_common_tag.get(word, fallback_tag)
            for word in words
        ]
        predicted_tags_per_sentence.append(predicted_tags)
        gold_tags_per_sentence.append(tags)

    return predicted_tags_per_sentence, gold_tags_per_sentence
    
if __name__ == "__main__":
    english_train_set = load_conllu_file("en_ewt-ud-train.conllu")
    basque_test_set = load_conllu_file("eu_bdt-ud-test.conllu")

    word_to_tag_en, fallback_en = train_frequency_baseline(english_train_set)
    predictions_zero_shot, gold_zero_shot = predict_with_frequency_baseline(
        basque_test_set, word_to_tag_en, fallback_en
    )

    print("Baseline (Zero-shot: English-trained, Basque-tested):")
    print("Precision:", precision_score(gold_zero_shot, predictions_zero_shot))
    print("Recall:", recall_score(gold_zero_shot, predictions_zero_shot))
    print("F1:", f1_score(gold_zero_shot, predictions_zero_shot))
    print("Accuracy:", accuracy_score(gold_zero_shot, predictions_zero_shot))

    basque_train_set = load_conllu_file("eu_bdt-ud-train.conllu")

    word_to_tag_eu, fallback_eu = train_frequency_baseline(basque_train_set)
    predictions_in_language, gold_in_language = predict_with_frequency_baseline(
        basque_test_set, word_to_tag_eu, fallback_eu
    )

    print("Baseline (In-language: Basque-trained, Basque-tested):")
    print("Precision:", precision_score(gold_in_language, predictions_in_language))
    print("Recall:", recall_score(gold_in_language, predictions_in_language))
    print("F1:", f1_score(gold_in_language, predictions_in_language))
    print("Accuracy:", accuracy_score(gold_in_language, predictions_in_language))
