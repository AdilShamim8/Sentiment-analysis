import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import emoji

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #4A90E2;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #357abd;
        transform: translateY(-2px);
    }
    .css-1d391kg {
        padding: 2rem;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 500;
    }
    .positive {
        background-color: rgba(46, 204, 113, 0.1);
        color: #2ecc71;
    }
    .negative {
        background-color: rgba(231, 76, 60, 0.1);
        color: #e74c3c;
    }
    .neutral {
        background-color: rgba(149, 165, 166, 0.1);
        color: #95a5a6;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('sentiment_analysis_model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keeping the text after #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """Full preprocessing pipeline"""
    if not isinstance(text, str):
        return ""
        
    # Clean text
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

def analyze_sentiment(text):
    """Analyze sentiment and return results"""
    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]
    return prediction, processed_text

def get_emoji(sentiment):
    """Get emoji based on sentiment"""
    if sentiment == 4:
        return "😊"
    elif sentiment == 0:
        return "😞"
    return "😐"

def plot_word_frequency(text):
    """Plot word frequency distribution"""
    words = text.split()
    word_freq = Counter(words)
    df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency'])
    df = df.sort_values('frequency', ascending=False).head(10)
    
    fig = px.bar(df, x='word', y='frequency',
                 title='Top 10 Most Frequent Words',
                 color='frequency',
                 color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis_title="Words",
        yaxis_title="Frequency",
        template="plotly_white"
    )
    return fig

def main():
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app uses a machine learning model to analyze the sentiment of text.
        It can classify text as positive, negative, or neutral.
        
        The model was trained on a large dataset of text samples and uses
        advanced natural language processing techniques.
        """
    )
    
    st.sidebar.title("Example Texts")
    example_texts = {
        "Positive": "I absolutely love this product! It's amazing and works perfectly.",
        "Negative": "This is the worst experience ever. I'm very disappointed.",
        "Neutral": "The product arrived on time and seems to be working as expected."
    }
    
    for sentiment, text in example_texts.items():
        if st.sidebar.button(f"{sentiment} Example"):
            st.session_state.text_input = text

    # Main content
    st.title("🧠 Sentiment Analysis")
    st.markdown("Analyze the sentiment of your text using advanced AI")
    
    # Text input
    text_input = st.text_area(
        "Enter your text here:",
        value=st.session_state.get('text_input', ''),
        height=150
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", use_container_width=True):
        if text_input:
            with st.spinner("Analyzing..."):
                # Get prediction
                sentiment, processed_text = analyze_sentiment(text_input)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Analysis Result")
                    sentiment_emoji = get_emoji(sentiment)
                    sentiment_text = "Positive" if sentiment == 4 else "Negative" if sentiment == 0 else "Neutral"
                    st.markdown(
                        f"""
                        <div class="sentiment-box {sentiment_text.lower()}">
                            {sentiment_emoji} {sentiment_text} Sentiment
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown("### Text Analysis")
                    st.markdown("**Original Text:**")
                    st.write(text_input)
                    st.markdown("**Processed Text:**")
                    st.write(processed_text)
                
                # Word frequency analysis
                st.markdown("### Word Frequency Analysis")
                fig = plot_word_frequency(processed_text)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional statistics
                st.markdown("### Text Statistics")
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric("Word Count", len(processed_text.split()))
                with col4:
                    st.metric("Character Count", len(text_input))
                with col5:
                    st.metric("Emoji Count", len([c for c in text_input if c in emoji.EMOJI_DATA]))
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    model = load_model()
    if model is not None:
        main()
    else:
        st.error("Failed to load the model. Please check if the model file exists.")