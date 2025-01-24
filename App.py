import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
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
st.markdown("<h3 style='color:blue;'>Use this tool to summarize relevant responses</h3>", unsafe_allow_html=True)

st.write("Please enter the keywords:")
col1, col2 = st.columns(2)

with col1:
    keyword1 = st.text_input("Keyword 1")

with col2:
    keyword2 = st.text_input("Keyword 2")

# Submission Summary
st.subheader("Submission Summary")
output = st.empty()

# Process User Input
if st.button("Generate a new submission"):
    if keyword1 or keyword2:
        try:
            user_keywords = [kw.strip() for kw in [keyword1, keyword2] if kw.strip()]
            
            # Debug: Show entered keywords
            st.write(f"Entered keywords: {user_keywords}")

            # Generate Embeddings for User Keywords
            keyword_embeddings = model.encode(user_keywords, convert_to_tensor=True)
            keyword_embeddings_cpu = normalize(keyword_embeddings.cpu().numpy(), axis=1)

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

            # Calculate Cosine Similarity for User Keywords
            response_similarities = {}
            for response_id, embedding in response_embeddings.items():
                response_similarities[response_id] = calculate_cosine_similarity(keyword_embeddings_cpu, {response_id: embedding})[response_id]

            # Dynamic Threshold
            similarity_scores = list(response_similarities.values())
            dynamic_threshold = np.percentile(similarity_scores, 85)

            # Filter Responses Based on Threshold
            filtered_responses = {
                response_id: sim for response_id, sim in response_similarities.items() if sim >= dynamic_threshold
            }

            TOP_N = 5
            if not filtered_responses:
                filtered_responses = dict(
                    sorted(response_similarities.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
                )

            filtered_texts = [
                " ".join(df_cleaned.loc[df_cleaned['ResponseID'] == resp_id, 'Responses'].values[0])
                for resp_id in filtered_responses.keys()
            ]

            if not filtered_texts:
                filtered_texts = ["No relevant responses found."]

            # Debug: Show filtered responses
            st.write(f"Filtered Responses: {filtered_texts}")

            # Prepare Input for T5 Summarization
            input_text = "summarize: " + " ".join(filtered_texts)
            input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True)

            # Generate Summary
            summary_ids = t5_model.generate(input_ids, max_length=200, num_beams=8, length_penalty=1.5, early_stopping=True)
            summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Display the Summary
            output.subheader("Generated Summary")
            output.text(summary)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    else:
        output.text("Please enter at least one keyword to generate a submission.")
