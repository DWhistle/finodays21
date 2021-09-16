# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# tokenizer = AutoTokenizer.from_pretrained("sismetanin/xlm_roberta_large-ru-sentiment-rureviews")

# model = AutoModelForSequenceClassification.from_pretrained("sismetanin/xlm_roberta_large-ru-sentiment-rureviews")
# input = tokenizer('Сколько можно ждать', return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

# print(model(**input, labels=labels))

# from transformers import AutoTokenizer, AutoModel
  
# tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")

# model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")

# input = tokenizer('Сколько можно ждать', return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

# print(model(**input))



import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)

@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    # predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted

print(predict("Ты мне нравишься. Я тебя люблю"))