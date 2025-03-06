import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification



def GPT_3():
    import torch
    from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class GPT3ForSequenceClassification(nn.Module):
        def __init__(self, gpt3_model, num_labels):
            super(GPT3ForSequenceClassification, self).__init__()
            self.gpt3 = gpt3_model
            self.classifier = nn.Linear(gpt3_model.config.n_embd, num_labels)  # num_labels is the number of classes

        def forward(self, input_ids, attention_mask=None):
            outputs = self.gpt3(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[0][:, 0, :]  # take the representation of the [CLS] token
            logits = self.classifier(pooled_output)
            return logits

    def train_model(model, dataloader, optimizer, criterion, epochs):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    # Example usage
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    gpt3_model = GPT2Model.from_pretrained("gpt2", config=config)

    # Modify GPT-3 for sequence classification
    model = GPT3ForSequenceClassification(gpt3_model, num_labels=1)

    # Example data
    texts = ["This is a positive review.", "This is a negative review."]
    labels = torch.tensor([1, 0])  # 1 for positive, 0 for negative

    # Tokenize input texts
    inputs = tokenizer(texts, padding=False, truncation=True, return_tensors="pt")

    # Create DataLoader
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Define optimizer, criterion, and train the model
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    epochs = 3
    train_model(model, dataloader, optimizer, criterion, epochs)
    return train_model



def Model_GPT_2():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

    # Example data
    texts = ["This is a positive review.", "This is a negative review."]
    labels = [1, 0]  # 1 for positive, 0 for negative

    # Tokenize input texts
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Convert labels to torch tensors
    labels = torch.tensor(labels)

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    epochs = 3
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(**tokenized_texts, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_texts)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
        print("Predictions:", predictions)
    return predictions


from transformers import OpenAIGPTTokenizer, TFOpenAIGPTModel
def Model_GPT_3(text):

    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    model = TFOpenAIGPTModel.from_pretrained("openai-gpt")

    inputs = tokenizer(text, return_tensors="tf")  # "Hello, my dog is cute"
    outputs = model(inputs)
    last_hidden_states = outputs.last_hidden_state
    Feature = np.reshape(last_hidden_states, last_hidden_states.shape[0] * last_hidden_states.shape[1] * last_hidden_states.shape[2])
    return Feature


import numpy as np


if __name__ == '__main__':
    prep = np.load('Preprocess.npy', allow_pickle=True)[:10]
    Feat = Model_GPT_3(prep)