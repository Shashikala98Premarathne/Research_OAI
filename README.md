# Research_OpenAI
Automation of Open Government Processes using Large Language Models. 



Abstractive Summarization Process Flow: 

1. Loading data in to a Pandas dataframe.
2. Data Preprocessing.
3. BERT Tokenization.
4. Response Embeddings Creation.
5. Grouping Response Embeddings by ResponseID using Mean-Pooling. 
6. User Input (Key-words).
7. Key-words Embedding creation.
8. Calculate and Compare Cosine Similarity Scores of Response embeddings and Key-words embeddings. 
8. Performing Grid-Search Thresholding method to optimize the Fiteration Process.
9. Optimal Threshold Selected.
10. Generate the Abstractive Summary. 