import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Academic Journal Recommender",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #f0f2f6;
    }
    .novelty-score {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .high-novelty {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-novelty {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-novelty {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

def string_to_array(embedding_str):
    """Convert string representation of embedding to numpy array"""
    try:
        cleaned_str = embedding_str.strip('[]').replace('\n', ' ')
        numbers = [float(x) for x in cleaned_str.split() if x]
        return np.array(numbers)
    except:
        return None

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def load_journal_embeddings(file_path='journal_embeddings.npz'):
    try:
        data = np.load(file_path, allow_pickle=True)
        journal_embeddings = {}
        for journal, embedding in zip(data['journals'], data['embeddings']):
            journal_embeddings[journal] = embedding
        return journal_embeddings
    except Exception as e:
        st.error(f"Error loading journal embeddings: {str(e)}")
        return None

@st.cache_data
def load_paper_database():
    try:
        return pd.read_csv('updated_dataframe.csv')
    except Exception as e:
        st.error(f"Error loading paper database: {str(e)}")
        return None

def calculate_novelty_score(paper_embedding, df, top_journals):
    """Calculate novelty score for the paper"""
    try:
        # Get papers from top journals
        relevant_papers = df[df['journal'].isin(top_journals)]
        
        # Calculate similarities with papers in top journals
        similarities_list = []
        for _, row in relevant_papers.iterrows():
            other_embedding = string_to_array(row['embedding'])
            if other_embedding is not None:
                other_embedding = other_embedding / np.linalg.norm(other_embedding)
                similarity = np.dot(paper_embedding, other_embedding)
                similarities_list.append(similarity)
        
        # Calculate average similarity (lower means more novel)
        avg_similarity = np.mean(similarities_list) if similarities_list else 0
        
        # Define novelty categories
        if avg_similarity < 0.1:
            category = "Highly Novel"
            score = 3
            css_class = "high-novelty"
        elif avg_similarity < 0.2:
            category = "Novel"
            score = 2
            css_class = "high-novelty"
        elif avg_similarity < 0.3:
            category = "Moderately Novel"
            score = 1
            css_class = "medium-novelty"
        else:
            category = "Not Novel"
            score = 0
            css_class = "low-novelty"
                
        return {
            'novelty_score': score,
            'novelty_category': category,
            'average_similarity': avg_similarity,
            'top_journals': top_journals,
            'css_class': css_class,
            'analyzed_papers': len(similarities_list)
        }
    except Exception as e:
        st.error(f"Error calculating novelty score: {str(e)}")
        return None

def embed_text(text, tokenizer, model):
    """Generate embedding for input text"""
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get text embedding
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def get_journal_recommendations(paper_embedding, journal_embeddings, top_k=5):
    """Get journal recommendations based on cosine similarity"""
    try:
        journal_names = list(journal_embeddings.keys())
        journal_emb_matrix = np.stack([journal_embeddings[j] for j in journal_names])
        similarities = np.dot(paper_embedding, journal_emb_matrix.T)[0]
        top_k_indices = np.argsort(-similarities)[:top_k]
        recommendations = [
            (journal_names[idx], similarities[idx]) 
            for idx in top_k_indices
        ]
        return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

def main():
    st.title("📚 Academic Journal Recommender")
    st.write("""
    Enter your paper's title and abstract to get personalized journal recommendations 
    and novelty analysis.
    """)
    
    # Load all required resources
    tokenizer, model = load_model()
    journal_embeddings = load_journal_embeddings()
    paper_db = load_paper_database()
    
    # Check each component separately
    if tokenizer is None or model is None:
        st.error("Failed to load the model. Please check the error messages above.")
        return
    if journal_embeddings is None:
        st.error("Failed to load journal embeddings. Please check the error messages above.")
        return
    if paper_db is None or paper_db.empty:
        st.error("Failed to load paper database or database is empty.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Paper Details")
        title = st.text_input("Paper Title")
        abstract = st.text_area("Abstract", height=200)
        
        num_recommendations = st.slider(
            "Number of recommendations", 
            min_value=1, 
            max_value=10, 
            value=5
        )
    
    with col2:
        st.subheader("Information")
        st.info("""
        💡 **Tips for best results:**
        - Provide a clear, detailed abstract
        - Include key methodology terms
        - Mention your research field
        - Include your main findings
        """)
    
    if st.button("Get Journal Recommendations", type="primary"):
        if not abstract.strip():
            st.warning("Please enter your paper's abstract.")
            return
        
        with st.spinner("Analyzing your paper..."):
            # Generate embedding
            text_to_analyze = f"{title}\n{abstract}" if title else abstract
            paper_embedding = embed_text(text_to_analyze, tokenizer, model)
            
            if paper_embedding is not None:
                # Get recommendations first
                recommendations = get_journal_recommendations(
                    paper_embedding, 
                    journal_embeddings, 
                    num_recommendations
                )
                
                if recommendations:
                    # Extract journal names
                    top_journals = [journal for journal, _ in recommendations]
                    
                    # Calculate novelty score using the top journals
                    novelty_results = calculate_novelty_score(
                        paper_embedding, 
                        paper_db,
                        top_journals
                    )
                
                if novelty_results and recommendations:
                    # Define the message based on the novelty category
                    if novelty_results['novelty_category'] == "Highly Novel":
                        message = "Your paper is novel enough to get published in Q1 journals/conferences."
                    elif novelty_results['novelty_category'] == "Novel":
                        message = "Your paper is novel enough to get published in Q2 journals/conferences."
                    elif novelty_results['novelty_category'] == "Moderately Novel":
                        message = "Your paper is novel enough to get published in Q3 journals/conferences."
                    else:
                        message = "Your paper may be more suitable for Q4 journals/conferences and may require additional novelty for higher impact."

                    # Display the novelty score with the custom message
                      # <p>Novelty Score: {novelty_results['novelty_score']}/3</p>
                        # <p>Average Similarity: {novelty_results['average_similarity']:.4f}</p>
                    st.markdown(f"""
                    <div class="novelty-score {novelty_results['css_class']}">
                        <h2>{novelty_results['novelty_category']}</h2>
                        <p><strong>{message}</strong></p>
                        <p>Based on analysis of {novelty_results['analyzed_papers']} papers from top recommended journals</p>
                        
                    </div>
                    """, unsafe_allow_html=True)


                    
                    st.subheader("📊 Journal Recommendations")
                    
                    # Display recommendations
                    for i, (journal, score) in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-box">
                                <h3>#{i}: {journal}</h3>
                                <p>Similarity Score: {score:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Add export functionality
                    results_df = pd.DataFrame({
                        'Rank': range(1, len(recommendations) + 1),
                        'Journal': [r[0] for r in recommendations],
                        'Similarity Score': [r[1] for r in recommendations],
                        'Novelty Category': [novelty_results['novelty_category']] * len(recommendations),
                        'Novelty Score': [novelty_results['novelty_score']] * len(recommendations),
                        'Average Similarity': [novelty_results['average_similarity']] * len(recommendations)
                    })
                    
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "journal_recommendations.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.error("Could not generate analysis. Please try again.")

if __name__ == "__main__":
    main()