from datasets import load_from_disk, disable_caching
from torch.utils.data import DataLoader
disable_caching()

IGNORE_TOKEN = -100

#####################
# FORMAT DATA       #
#####################

template_context = """### Instruction:
{instruction}

### Context:
{context}

### Response:
"""

template_no_context = """### Instruction:
{instruction}

### Response:
"""

def data_to_string(data):

    instruction = data["instruction"]
    context = data["context"]
    response = data["response"]

    template = template_context if len(context) > 0 else template_no_context

    source = template.format(instruction=instruction, context=context)

    return {
        "source": source,
        "text": source + response, 
    }

original_dataset = load_from_disk("../datasets/databricks-dolly-15k")["train"] 

dataset = original_dataset.map(data_to_string).remove_columns(original_dataset.column_names)

# filter dataset to exclude examples that are too long
dataset = dataset.filter(lambda x: len(x["text"]) < 500)

#####################
# SPLIT DATA        #
#####################

processed_dataset = dataset.train_test_split(test_size=0.1)

train_dataset = processed_dataset["train"]
eval_dataset = processed_dataset["test"]

#####################
# CREATE DATALOADER #
#####################


def data_collator(features, tokenizer):
    sources = [feature["source"] for feature in features]
    targets = [feature["text"] for feature in features]

    source_tokens = tokenizer(
        sources,
        return_tensors="pt",
        padding='longest',
        max_length=None,
    )

    target_tokens = tokenizer(
        targets,
        return_tensors="pt",
        padding='longest',
        max_length=None,
    )
        
    labels = target_tokens["input_ids"].clone()

    for i in range(len(labels)):
        source_len = source_tokens["attention_mask"][i].sum()
        
        labels[i, :source_len] = IGNORE_TOKEN

    res = {
        "input_ids": target_tokens["input_ids"],
        "attention_mask": target_tokens["attention_mask"],
        "labels": labels,
    }

    return res

def load_data(tokenizer, batch_size):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: data_collator(x, tokenizer))

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: data_collator(x, tokenizer))

    return train_dataloader, eval_dataloader
