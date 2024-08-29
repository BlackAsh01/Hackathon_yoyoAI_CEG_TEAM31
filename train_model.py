import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Step 1: Load the labeled data from the JSON file
json_file_path = 'analysis_final.json'  # Replace with your actual file path
with open(json_file_path, 'r') as f:
    labeled_data = json.load(f)

# Step 2: Flatten the data
def flatten_data(data):
    flat_data = []
    for entry in data:
        flattened_entry = {
            "conversation": entry,  # Storing the full entry for tokenization
            "budget": entry['customer_requirements'].get('budget', 0),
            "car_type": entry['customer_requirements'].get('car_type', "unknown"),
            "color": entry['customer_requirements'].get('color', "unknown"),
            "fuel_type": entry['customer_requirements'].get('fuel_type', "unknown"),
            "transmission_type": entry['customer_requirements'].get('transmission_type', "unknown"),
            "free_rc_transfer": entry['company_policies'].get('free_rc_transfer', False),
            "money_back_guarantee": entry['company_policies'].get('money_back_guarantee', False),
            "free_rsa": entry['company_policies'].get('free_rsa', False),
            "return_policy": entry['company_policies'].get('return_policy', False),
            "refurbishment_quality": bool(entry['customer_objections'].get('refurbishment_quality')),
            "car_issues": bool(entry['customer_objections'].get('car_issues')),
            "price_issues": bool(entry['customer_objections'].get('price_issues')),
            "customer_experience_issues": bool(entry['customer_objections'].get('customer_experience_issues')),
        }
        flat_data.append(flattened_entry)
    return flat_data

flat_data = flatten_data(labeled_data)

# Step 3: Define the Dataset class
class CarSalesDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = json.dumps(self.data[idx]['conversation'])
        labels = [
            int(self.data[idx].get('budget', 0)),
            self.data[idx].get('car_type', "unknown"),
            self.data[idx].get('color', "unknown"),
            self.data[idx].get('fuel_type', "unknown"),
            self.data[idx].get('transmission_type', "unknown"),
            int(self.data[idx].get('free_rc_transfer', False)),
            int(self.data[idx].get('money_back_guarantee', False)),
            int(self.data[idx].get('free_rsa', False)),
            int(self.data[idx].get('return_policy', False)),
            int(self.data[idx].get('refurbishment_quality', False)),
            int(self.data[idx].get('car_issues', False)),
            int(self.data[idx].get('price_issues', False)),
            int(self.data[idx].get('customer_experience_issues', False)),
        ]

        # Tokenize the conversation
        encoding = self.tokenizer.encode_plus(
            conversation,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Convert labels to tensor
        label_tensor = torch.tensor(labels, dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }

# Step 4: Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Step 5: Create the dataset
dataset = CarSalesDataset(flat_data, tokenizer, max_len=128)

# Step 6: Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Step 7: Initialize the model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(flat_data[0].keys()) - 1)

# Step 8: Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Step 9: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Step 10: Train the model
trainer.train()

# Step 11: Save the trained model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
