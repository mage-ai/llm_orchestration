from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example text
# text = "This is an example sentence."

# # Tokenize the text into subword tokens
# encoded = tokenizer.encode_plus(text, return_tensors='pt')

# # Get the subword token ids
# token_ids = encoded['input_ids']

# # Get the attention mask (required for BERT)
# attention_mask = encoded['attention_mask']

# # Feed the token ids and attention mask to the BERT model
# with torch.no_grad():
#     outputs = model(token_ids, attention_mask=attention_mask)

# # Get the embeddings for each subword token
# embeddings = outputs.last_hidden_state