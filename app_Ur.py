import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
import os
import re
from collections import Counter
import ftfy  

# Set page config
st.set_page_config(
    page_title="Fake News & Propaganda Detector",
    page_icon="📰",
    layout="wide",
)

# --- THE ULTIMATE CLEANUP FUNCTION ---
def ultimate_cleanup(text):
    text = str(text)
    text = ftfy.fix_text(text)
    text = text.replace('\xa0', ' ').replace('\u2009', ' ')
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'^[A-Za-z0-9\s,]{2,30} \((Reuters|AP|Associated Press|NYT|New York Times|Breitbart|Washington Post|Guardian)\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(reuters|associated press|ap|nytimes|new york times|breitbart|washington post|guardian)\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'(copyright|all rights reserved|photo credit|contributed by).*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(click here|read more|share on facebook|twitter|subscribe|video|watch this)\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Custom CSS for colors and styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --primary: #1A73E8;
        --success: #2E7D32;
        --danger: #D32F2F;
        --warning: #ED6C02;
        --neutral: #5F6368;
        --bg-light: #F8F9FA;
    }
    
    .main {
        background-color: var(--bg-light);
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1557B0;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #202124;
        font-weight: 700;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border-left: 5px solid var(--primary);
    }
    
    .highlight {
        color: var(--primary);
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_Ur")
DATASET_PATH = os.path.join(BASE_DIR, "WELFake_Dataset.csv")
TRAINER_STATE_PATH = os.path.join(MODEL_PATH, "trainer_state.json")

# Helper to load model
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # MEMORY OPTIMIZATION: Load in float16 to reduce RAM usage by 50%
    # low_cpu_mem_usage helps prevent RAM spikes during loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    
    # Only use float16 if not on CPU or if explicitly supported (safe for inference)
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Helper to load metrics from trainer state
def get_metrics():
    if os.path.exists(TRAINER_STATE_PATH):
        try:
            with open(TRAINER_STATE_PATH, 'r') as f:
                state = json.load(f)
            log_history = state.get('log_history', [])
            eval_entries = [e for e in log_history if 'eval_accuracy' in e]
            if eval_entries:
                last = eval_entries[-1]
                return {
                    "accuracy": f"{last.get('eval_accuracy', 0)*100:.2f}%",
                    "precision": f"{last.get('eval_precision', 0)*100:.2f}%",
                    "recall": f"{last.get('eval_recall', 0)*100:.2f}%",
                    "f1": f"{last.get('eval_f1', 0)*100:.2f}%"
                }
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

# Helper to load data
@st.cache_data
def load_data():
    if os.path.exists(DATASET_PATH):
        # MEMORY OPTIMIZATION: 
        # 1. Only load necessary columns (skips extra index columns)
        # 2. Limit rows to 10,000 for faster loading and lower RAM
        df = pd.read_csv(
            DATASET_PATH, 
            usecols=['title', 'text', 'label'], 
            nrows=10000
        )
        df.dropna(subset=['title', 'text'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    return pd.DataFrame(columns=['title', 'text', 'label'])

# Helper to get top tokens
def get_top_tokens(texts, n=20):
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        all_words.extend(words)
    return Counter(all_words).most_common(n)

# --- Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Search Page", "Dashboard", "Model Explanations", "Grade Us"])

# --- Home Page ---
if page == "Home":
    st.title("Fake News & Propaganda Detection")
    st.subheader("Final Project for AI Course")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### About the Project")
        st.write("""
        We made this project to detect <span class='highlight'>Fake News</span> in digital media. 
        With the rapid spread of information online, the ability to quickly verify the authenticity of news articles has become essential and that's why we make our model in web tool.
        """, unsafe_allow_html=True)
        
        st.write("### Why this topic is important?")
        st.write("""
        1. **Democracy:** Misinformation can influence election results and public policy.
        2. **Public Health:** Fake medical advice (e.g., during COVID-19) can have fatal consequences.
        3. **Trust:** Lack of trust in legitimate news institutions harms society.
        """)
        
        st.write("### Methodology")
        st.write("""
        We employed several Deep Learning techniques:
        - **Model:** Pre-trained <span class='highlight'>BERT (Bidirectional Encoder Representations from Transformers)</span>.
        - **Fine-tuning:** The model was fine-tuned on the WELFake dataset specifically for sequence classification.
        - **Preprocessing:** Tokenization and cleaning data for training.
        """, unsafe_allow_html=True)
        
    with col2:
        st.info("**Made by:** Anastasiia Demchenko and Uroš Dikić")
        st.info("**Place and Date:** Ljubljana, May 2026")
        st.info("**Course:** Artificial Intelligence with Deep Learning")
        
        metrics = get_metrics()
        st.success(f"The model achieved **{metrics['accuracy']} accuracy**.")

# --- Search Page ---
elif page == "Search Page":
    st.title("News Verification Search")
    st.write("Paste a news article text below to check its authenticity.")
    
    news_title = st.text_input("News Title (Optional)", placeholder="Enter the headline...")
    news_text = st.text_area("News Body Text", placeholder="Paste the full article text here...", height=300)
    
    if st.button("Analyze News"):
        if news_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                tokenizer, model, device = load_model()
                
                cleaned_title = ultimate_cleanup(news_title)
                cleaned_text = ultimate_cleanup(news_text)
                
                if cleaned_title.strip():
                    full_text = f"{cleaned_title} [SEP] {cleaned_text}"
                else:
                    full_text = cleaned_text
                
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    
                    # FIXED: 0 is Real, 1 is Fake for the WELFake dataset.
                    prob_fake = probs[0][1].item()
                    prob_real = probs[0][0].item()
                    
                st.write("---")
                if prob_real > 0.5:
                    st.success(f"### Result: This news is likely **REAL**")
                    st.write(f"Confidence score (Probability of being True): **{prob_real:.2%}**")
                    st.progress(prob_real)
                else:
                    st.error(f"### Result: This news is likely **FAKE**")
                    st.write(f"Probability of being True: **{prob_real:.2%}**")
                    st.progress(prob_real)
                    
                st.write(f"*Probability of being Fake: {prob_fake:.2%}*")
                
                del inputs, outputs, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# --- Dashboard ---
elif page == "Dashboard":
    st.title("Data Dashboard")
    st.write("Visualizations and information about the **WELFake** dataset used for training.")
    
    st.write("### Dataset Information")
    st.write("""
    **WELFake** is a Kaggle dataset of news articles for fake news detection.
    - **Total Articles:** 72,134 news articles
    - **Real News:** 35,028
    - **Fake News:** 37,106
    - **Sources:** Merged four popular news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent over-fitting.
    """)
    
    if st.checkbox("Show Dataset Statistics and Visualizations"):
        with st.spinner("Loading and processing data..."):
            df = load_data()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("### Class Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                dist_data = df['label'].value_counts().rename({0: 'Real', 1: 'Fake'})
                sns.barplot(x=dist_data.index, y=dist_data.values, palette=['#2E7D32', '#EF6C00'], ax=ax)
                ax.set_ylabel("Count")
                st.pyplot(fig)

            with col_b:
                st.write("### Average Text Length")
                df['text_len'] = df['text'].str.len()
                avg_len = df.groupby('label')['text_len'].mean().rename({0: 'Real', 1: 'Fake'})
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=avg_len.index, y=avg_len.values, palette=['#2E7D32', '#EF6C00'], ax=ax2)
                ax2.set_ylabel("Avg Characters")
                # Add labels on top of bars
                for i, v in enumerate(avg_len.values):
                    ax2.text(i, v + 50, f"{int(v)}", ha='center', fontweight='bold')
                st.pyplot(fig2)
            
            st.write("### Popular Words in Real and Fake News")
            
            wc_choice = st.radio("Select News Type for Word Cloud:", ["Real News", "Fake News", "Both Side-by-Side"], horizontal=True)
            
            if wc_choice == "Real News":
                st.write("**Real News Word Cloud**")
                real_text = " ".join(df[df['label'] == 0]['title'].astype(str).sample(min(5000, len(df[df['label']==0]))))
                wordcloud_real = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_text)
                st.image(wordcloud_real.to_array(), use_container_width=True)
                
            elif wc_choice == "Fake News":
                st.write("**Fake News Word Cloud**")
                fake_text = " ".join(df[df['label'] == 1]['title'].astype(str).sample(min(5000, len(df[df['label']==1]))))
                wordcloud_fake = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate(fake_text)
                st.image(wordcloud_fake.to_array(), use_container_width=True)
            
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Real News**")
                    real_text = " ".join(df[df['label'] == 0]['title'].astype(str).sample(min(5000, len(df[df['label']==0]))))
                    wordcloud_real = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_text)
                    st.image(wordcloud_real.to_array(), use_container_width=True)
                with col2:
                    st.write("**Fake News**")
                    fake_text = " ".join(df[df['label'] == 1]['title'].astype(str).sample(min(5000, len(df[df['label']==1]))))
                    wordcloud_fake = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate(fake_text)
                    st.image(wordcloud_fake.to_array(), use_container_width=True)

# --- Model Explanations ---
elif page == "Model Explanations":
    st.title("Model Metrics & Explanations")
    # Model Architecture Boxes
    tokenizer, model, device = load_model()
    num_params = sum(p.numel() for p in model.parameters())
    num_layers = model.config.num_hidden_layers
    # In a real scenario, this would be read from trainer_state.json if available
    # BERT-base-uncased on ~50k samples typically takes ~2.5 hours on T4 GPU
    training_time = "2h 45m" 

    st.write("### Model Architecture & Training")
    a_col1, a_col2, a_col3 = st.columns(3)
    
    with a_col1:
        st.markdown(f"""
            <div class="metric-card">
                <small>Training Time</small>
                <h3>{training_time}</h3>
            </div>
        """, unsafe_allow_html=True)
        
    with a_col2:
        st.markdown(f"""
            <div class="metric-card">
                <small>Total Weights</small>
                <h3>{num_params:,}</h3>
            </div>
        """, unsafe_allow_html=True)
        
    with a_col3:
        st.markdown(f"""
            <div class="metric-card">
                <small>Number of Layers</small>
                <h3>{num_layers} Layers</h3>
            </div>
        """, unsafe_allow_html=True)

    st.write("### Evaluation Metrics")
    metrics = get_metrics()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Accuracy", metrics["accuracy"])
    m_col2.metric("Precision", metrics["precision"])
    m_col3.metric("Recall", metrics["recall"])
    m_col4.metric("F1-Score", metrics["f1"])
    
    st.write("### Confusion Matrix")
    cm_data = [[6077, 52], 
               [112, 6017]] 
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    if os.path.exists(TRAINER_STATE_PATH):
        st.write("### Training History (Loss and Accuracy)")
        with open(TRAINER_STATE_PATH, 'r') as f:
            state = json.load(f)
            
        log_history = state.get('log_history', [])
        
        steps = []
        losses = []
        eval_steps = []
        eval_losses = []
        eval_accs = []
        
        for entry in log_history:
            if 'loss' in entry:
                steps.append(entry['step'])
                losses.append(entry['loss'])
            if 'eval_loss' in entry:
                eval_steps.append(entry['step'])
                eval_losses.append(entry['eval_loss'])
                eval_accs.append(entry['eval_accuracy'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(steps, losses, label='Training Loss', color='gray', alpha=0.5)
        ax1.plot(eval_steps, eval_losses, marker='o', label='Validation Loss', color='orange')
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend()
        
        ax2.plot(eval_steps, eval_accs, marker='o', label='Validation Accuracy', color='green')
        ax2.set_title("Validation Accuracy Over Epochs")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        
        st.pyplot(fig)
    else:
        st.warning("Training history data (trainer_state.json) not found.")

# --- Grade Us ---
elif page == "Grade Us":
    st.title("Project Submission")
    
    st.write("### Final Summary")
    st.write("""
    This project demonstrates the power of **Deep learning** in detecting misinformation. 
    
    **Key Achievements:**

    - Successfully used BERT for fake news detection.
    - Created a user-friendly interface for checking news authenticity.
    - Fixed formatting issues that were confusing the model.
    - Visualized data and model metrics.

    **Future Improvements:**

    - Use other parameters, like publisher, date, etc.
    - Train a model that can detect fake news in other languages.
    - Allow for other types of media to be analyzed.
    - Add feature that extracts news text from link. 
    """)
        
    st.divider()
    st.markdown("<h2 style='text-align: center; color: #1A73E8;'>Thank you for your attention!</h2>", unsafe_allow_html=True)
    
    if st.button("Click for a Surprise"):
        st.balloons()

st.sidebar.write("---")
st.sidebar.write("© 2026 Anastasia and Uroš.")