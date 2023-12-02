import torch
import pandas as pd
from nltk.corpus import brown
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_brown_corpus_data():
    # Preparing data
    fiction = ["adventure", "fiction", "mystery", "romance", "science_fiction"]
    nonfiction = ["government", "hobbies", "learned", "news", "reviews"]

    fiction_ids = [x for y in fiction for x in brown.fileids(categories=y)]
    nonfiction_ids = [x for y in nonfiction for x in brown.fileids(categories=y)]

    data = []
    for index, fileid in enumerate(fiction_ids + nonfiction_ids):
        paras = brown.paras(fileids=fileid)
        label = 1 if fileid in fiction_ids else 0

        for j, p in enumerate(paras):
            if len(p) > 4 and len(p) < 7:
                text = ""
                for sen in p:
                    text = text + " " + " ".join(sen)
            temp = {}
            temp["text"] = text
            temp["label"] = label
            data.append(temp)
    df_brown = pd.DataFrame(data)
    return df_brown


# Preparing training arguments

device = torch.device("cuda")
print("Device: ", device)


class DataPreparation(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    cl = classification_report(labels, preds)
    return {"accuracy": acc, "report": cl}


def train_and_evaluate(df_train, df_test, model_name, save_model=False):
    # Repeating training with different arguments
    print(f"Using {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_texts = list(df_train.text)
    train_labels = list(df_train.label)

    test_texts = list(df_test.text)
    test_labels = list(df_test.label)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=512
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=512
    )

    train_dataset = DataPreparation(train_encodings, train_labels)
    test_dataset = DataPreparation(test_encodings, test_labels)

    model_short_name = "bert_base_uncased_fine_tuned_on_brown_corpus"

    args = TrainingArguments(
        f"../resources/artifacts/{model_short_name}",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_first_step=True,
        save_strategy=("epoch" if save_model else "no"),
        disable_tqdm=True,
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("Evaluating->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    print(trainer.evaluate())

    print("Clearing cuda cache")
    torch.cuda.empty_cache()

    return trainer


if __name__ == "main":
    # Getting data
    df_brown = get_brown_corpus_data()

    # model name
    model_name = "bert-base-uncased"
    for i in range(10):
        print(f"starting split {i} :")
        train, test = train_test_split(df_brown, test_size=0.3, random_state=i)
        trainer_model = train_and_evaluate(
            train, test, model_name=model_name, save_model=False
        )
        print(f"Testing Done on split {i} :")
