## Chunking, tokenization, and encoding

Chunking involves breaking down larger documents or pieces of text into smaller, more manageable segments called "chunks". 
The goal of chunking is to create chunks that are optimally sized for the specific RAG application, 
balancing factors like retrieval quality, storage costs, serving latency, and the limitations of the language model being used.

Tokenization is the process of breaking down raw text data into smaller units called tokens, which can be words, subwords, or even individual characters.
It is typically the first step in preparing text for input into machine learning models like RAG.
Tokenization allows the text to be converted into a numerical format that can be processed by the models.

Encoding or embedding is the conversion of these text units into vector representations. 