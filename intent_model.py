import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem.porter import PorterStemmer
import re
import os

FILE = "trained_chat_model.pth"

class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)
		return out

class ChatDataset(Dataset):
	def __init__(self):
		self.n_samples = len(X_train)
		self.x_data = X_train
		self.y_data = y_train

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.n_samples


# Helper functions for preprocessing
def tokenize(sentence):
	return sentence.split()

def stem(word):
	stemmer = PorterStemmer()
	return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
	sentence_words = [stem(w) for w in tokenized_sentence]
	bag = np.zeros(len(words), dtype=np.float32)
	for idx, w in enumerate(words):
		if w in sentence_words:
			bag[idx] = 1
	return bag

def load_model(model_file):
    data = torch.load(model_file)
    model_state = data["model_state"]
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    return model, all_words, tags

def predict(model, sentence, all_words, tags, device):
    model.to(device)
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

    return tag

# Check if the file exists in the current directory
if not os.path.exists(FILE):
	# Load dataset
	df = pd.read_csv('chatbot_intents_dataset_from_json.csv')  # Replace with the path to the downloaded dataset

	all_words = []
	tags = []
	xy = []

	# Process each pattern in the dataset
	ignore_words = ['?', '!', '.', ',']
	for _, row in df.iterrows():
		tag = row['Tag']
		pattern = row['Pattern']
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w, tag))
		if tag not in tags:
			tags.append(tag)

	# Filter out unwanted characters and stem words
	all_words = [stem(w) for w in all_words if w not in ignore_words]
	all_words = sorted(set(all_words))
	tags = sorted(set(tags))

	# Create training data
	X_train = []
	y_train = []
	for (pattern_sentence, tag) in xy:
		bag = bag_of_words(pattern_sentence, all_words)
		X_train.append(bag)
		label = tags.index(tag)
		y_train.append(label)

	# Convert to numpy arrays
	X_train = np.array(X_train)
	y_train = np.array(y_train)

	batch_size = 8
	hidden_size = 8
	output_size = len(tags)
	input_size = len(X_train[0])
	learning_rate = 0.001
	num_epochs = 500
	
	# Data loader
	train_loader = DataLoader(dataset=ChatDataset(), batch_size=batch_size, shuffle=True)
	
	# Model initialization
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = NeuralNet(input_size, hidden_size, output_size).to(device)
	
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	# Training loop
	for epoch in range(num_epochs):
		for (words, labels) in train_loader:
			words = words.to(device)
			labels = labels.to(dtype=torch.long).to(device)
			
			# Forward pass
			outputs = model(words)
			loss = criterion(outputs, labels)
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if (epoch + 1) % 100 == 0:
			print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

	data = {
		"model_state": model.state_dict(),
		"input_size": input_size,
		"hidden_size": hidden_size,
		"output_size": output_size,
		"all_words": all_words,
		"tags": tags
	}

	torch.save(data, FILE)

# Load the model and make a prediction
model, all_words, tags = load_model(FILE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)