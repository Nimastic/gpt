from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

def load_conversation_dataset(file_path):
    dataset = load_dataset('text', data_files={'train': file_path})
    return dataset['train']

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

def group_texts(examples):
    block_size = 128
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])

    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

dataset = load_conversation_dataset('conversation_data.txt')
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
lm_dataset = tokenized_dataset.map(group_texts, batched=True)

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
