from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.optim import AdamW
from accelerate import Accelerator
import time
from progress_bar import get_progress_bar
from get_data import load_data
import torch

accelerator = Accelerator()

####################
# CONSTANTS        #
####################

IGNORE_TOKEN = -100
MODEL_PATH = "/mnt/models/llama2/hf/Llama-2-7b-hf"
NAME = "llama2-dolly-15k"
BATCH_SIZE = 8
NUM_EPOCHS = 3

####################
# TOKENIZER + DATA #
####################

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, legacy=False)
tokenizer.pad_token = tokenizer.eos_token

train_dataloader, eval_dataloader = load_data(tokenizer, BATCH_SIZE)
num_batches = len(train_dataloader)

####################
# MODEL            #
####################

model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
model = accelerator.prepare(model)

####################
# OPTIMIZER + PREP #
####################

optimizer = AdamW(model.parameters(), lr=3e-5)

optimizer, train_dataloader, eval_dataloader = accelerator.prepare(optimizer, train_dataloader, eval_dataloader)

print("Prepared model to run with Accelerator")

# PROGRESS BAR
progress_bar = get_progress_bar(accelerator)

with progress_bar:
    task = progress_bar.add_task(
        "Training", total=num_batches*NUM_EPOCHS, loss=0.0, speed=0.0
    )

    loss_history = []

    def rolling_average(window):
        if len(loss_history) < window:
            return sum(loss_history) / len(loss_history)
        return sum(loss_history[-window:]) / window

    for epoch in range(NUM_EPOCHS):

        model.train() # set model to train mode
        total_loss = 0

        for idx, batch in enumerate(train_dataloader):
            start = time.time()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss_history.append(loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            progress_bar.update(
                task, advance=1, loss=loss.item(), speed=time.time() - start
            )

            if idx % 50 == 0:
                print(f"Avg. loss at epoch {epoch + idx/num_batches:.3f}: {rolling_average(50):.4f}")

                torch.cuda.empty_cache()

# TODO: model evaluate (remember to unset model to train mode)

# save model
model.save_pretrained(f"/mnt/finetunes/{NAME}")
tokenizer.save_pretrained(f"/mnt/finetunes/{NAME}")