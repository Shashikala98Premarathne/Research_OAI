import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import streamlit as st

# Load Preprocessed Data and Embeddings
try:
    df_cleaned = pd.read_csv("df_cleaned.csv")
    with open("response_embeddings.pkl", "rb") as f:
        response_embeddings = pickle.load(f)
except FileNotFoundError as e:
    st.error("Error: Missing required files. Ensure df_cleaned.csv and response_embeddings.pkl are available.")
    st.stop()

# Initialize Models
model = SentenceTransformer('all-mpnet-base-v2')
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Streamlit Web Interface
st.title("Open Government Consultation")
st.markdown("<h3 style='color:blue;'>Your Feedback is Highly Valued and Crucial in Shaping Better Governance</h3>", unsafe_allow_html=True)

# User Input for Keywords
st.write("Please enter the keywords:")
col1, col2 = st.columns(2)

with col1:
    keyword1 = st.text_input("Keyword 1")

with col2:
    keyword2 = st.text_input("Keyword 2")

# Function to calculate cosine similarity
def calculate_cosine_similarity(query_embeddings, response_embeddings):
    similarities = {}
    for response_id, embedding in response_embeddings.items():
        response_embedding_numpy = embedding.detach().cpu().numpy()
        response_embedding_normalized = normalize(response_embedding_numpy.reshape(1, -1), axis=1)
        max_similarity = max(
            cosine_similarity(query_embeddings, response_embedding_normalized).flatten()
        )
        similarities[response_id] = max_similarity
    return similarities

# Summarization 
if st.button("Generate a Summary"):
    if keyword1 or keyword2:
        try:
            user_keywords = [kw.strip() for kw in [keyword1, keyword2] if kw.strip()]

            # Generate Embeddings for User Keywords
            keyword_embeddings = model.encode(user_keywords, convert_to_tensor=True)
            keyword_embeddings_cpu = normalize(keyword_embeddings.detach().cpu().numpy(), axis=1)

            # Calculate Cosine Similarity between User Keywords and Responses
            response_similarities = {}
            for response_id, embedding in response_embeddings.items():
                response_similarities[response_id] = calculate_cosine_similarity(keyword_embeddings_cpu, {response_id: embedding})[response_id]

            # Grid-Search to Find the Best Threshold
            threshold_range = np.arange(0.0, 1.01, 0.01)
            best_threshold = None
            best_rouge1_fmeasure = -float("inf")
            filtered_responses_best = None

            for threshold in threshold_range:
                # Filter Responses Based on Threshold
                filtered_responses = {
                    response_id: sim for response_id, sim in response_similarities.items() if sim >= threshold
                }

                if not filtered_responses:
                    continue

                # Combine Filtered Responses into a Single String
                filtered_reference = " ".join([
                    df_cleaned.loc[df_cleaned['ResponseID'] == resp_id, 'Responses'].values[0]
                    for resp_id in filtered_responses.keys()
                ])

                # Prepare Input for T5 Summarization
                input_text = "summarize: " + filtered_reference
                input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True)

                # Generate Summary
                summary_ids = t5_model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
                summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Evaluate Using ROUGE-1 F-Measure
                rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
                rouge_scores = rouge_scorer_instance.score(filtered_reference, summary)
                rouge1_fmeasure = rouge_scores['rouge1'].fmeasure

                # Update Best Threshold
                if rouge1_fmeasure > best_rouge1_fmeasure:
                    best_rouge1_fmeasure = rouge1_fmeasure
                    best_threshold = threshold
                    filtered_responses_best = filtered_responses

            # Generate Final Summary Using Best Threshold
            if filtered_responses_best:
                final_reference = " ".join([
                    df_cleaned.loc[df_cleaned['ResponseID'] == resp_id, 'Responses'].values[0]
                    for resp_id in filtered_responses_best.keys()
                ])
                input_text = "summarize: " + final_reference
                input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
                summary_ids = t5_model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
                final_summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                st.success("Abstractive Summary:")
                st.write(final_summary)
            else:
                st.warning("No relevant responses found.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter at least one keyword.")
