# Import required modules
import asyncio
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from deep_translator import GoogleTranslator
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Dictionary of language codes and their names
LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-CN': 'Chinese (Simplified)',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'sw': 'Swahili'
}

# Tool definition for adding numbers
@tool
def add_two_numbers(a: int, b: int) -> str:
    """
    Adds two numbers together
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        str: The sum of the two numbers
    """
    # Convert result to string since LLM expects string output
    return str(a + b)

# Tool definition for searching vector database
@tool
def search_vector_db(query: str) -> str:
    """
    Search the vector database for documents similar to the query.
    Args:
        query (str): The search query string to find relevant documents
    Returns:
        str: A concatenated string of the top 5 most similar document contents found in the vector database
    """
    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Initialize/connect to vector database
    vector_store = Chroma(
        collection_name="embeddings",
        embedding_function=embeddings,
        persist_directory="./vector_db",
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # Debug print
    print("Searching the vector database for: ", query)
    
    # Perform similarity search and get top 5 results
    result = vector_store.similarity_search(query=query, k=5)
    # Combine all document contents into a single string
    result_str = "\n".join([doc.page_content for doc in result])
    
    return result_str


class TranslationManager:
    def __init__(self):
        self.translator = GoogleTranslator()
    
    @tool
    def translate_text(self, text: str, target_lang: str, source_lang: str = 'auto') -> str:
        """
        Translate text to target language
        """
        try:
            self.translator = GoogleTranslator(source=source_lang, target=target_lang)
            return self.translator.translate(text)
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text

# Main chat class that uses Gemini LLM
class GeminiChat:
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.0):
        """
        Initialize GeminiChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gemini-pro".
            temperature (float): The temperature to use. Default is 0.0.
        """
        # Store API key
        self.api_key = api_key
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature
        )
        
        # Create agent with both tools available
        self.agent = create_react_agent(self.llm, tools=[add_two_numbers, search_vector_db])
        
        # Initialize conversation history
        self.messages = []
        
        # Initialize translation manager
        self.translation_manager = TranslationManager()
        
    def send_message(self, message: str, source_lang: str, target_lang: str) -> str:
        """
        Send a message, translate it, get response, and translate back
        """
        # Translate user message to English (if not already in English)
        if source_lang != 'en':
            message = self.translation_manager.translate_text(message, 'en', source_lang)
        
        # Add user message to history
        self.messages.append(HumanMessage(content=message))
        
        # Store current history length
        history_length = len(self.messages)
        
        # Get response from agent
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        
        # Extract new messages
        new_messages = self.messages[history_length:]
        
        # Translate response back to target language (if not English)
        if target_lang != 'en':
            translated_messages = []
            for msg in new_messages:
                translated_content = self.translation_manager.translate_text(
                    msg.content, target_lang, 'en'
                )
                if isinstance(msg, AIMessage):
                    translated_messages.append(AIMessage(content=translated_content))
                else:
                    translated_messages.append(msg)
            return translated_messages
        
        return new_messages

# Streamlit interface with enhanced styling
st.set_page_config(
    page_title="Tiri Ya Tiyara",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics
def apply_custom_styling():
    st.markdown("""
    <style>
    /* Global Styling */
    .stApp {
        background-color: #F0F4F8;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }

    /* Sidebar Styling */
    .css-1aumxhk {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Title Styling */
    .stTitle {
        color: #2D3748;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #4299E1;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #3182CE;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Card Styling */
    .stCard {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #CBD5E0;
        padding: 10px;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4299E1;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.2);
    }

    /* Selectbox Styling */
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #CBD5E0;
        background-color: white;
    }

    /* Alert Styling */
    .stAlert {
        border-radius: 8px;
        opacity: 0.9;
    }

    /* Navigation Styling */
    .nav-item {
        padding: 10px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .nav-item:hover {
        background-color: #E2E8F0;
        transform: translateX(5px);
    }

    /* Gradient Background for Home */
    .home-gradient {
        background: linear-gradient(135deg, #4299E1 0%, #3182CE 100%);
        color: white;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
    }

    /* Translation-specific styling */
    .translation-container {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .language-select {
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Configuration
def configure_sidebar():
    with st.sidebar:
        # Enhanced Branding
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
            <img src="https://as1.ftcdn.net/v2/jpg/02/22/40/74/1000_F_222407479_CeUp7K5OSIV9erlj7KjbLZfDSMvpVrGX.jpg" 
                 style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover;">
            <h1 style="margin-left: 15px; color: #2D3748;">Tiri Ya Tiyara</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧭 Navigation", unsafe_allow_html=True)
        pages = ["Home", "Plan My Trip", "Saved Trips", "About Us"]
        
        # Custom Radio with hover effect
        page_selection = st.radio("Select Page", pages, label_visibility="collapsed")
        
        st.markdown("---")
        
        # Travel Preferences
        st.markdown("### 🌍 Travel Preferences")
        
        # Styled Containers
        with st.container():
            travel_type = st.selectbox("Travel Style", 
                ["Adventure", "Relaxation", "Cultural", "Luxury", "Budget"],
                key="travel_style")
            
        with st.container():
            budget = st.slider("Budget Range ($)", 
                min_value=500, 
                max_value=5000, 
                value=1500, 
                step=100)
            
        travel_dates = st.date_input("Travel Dates")
        
        return page_selection, travel_type, budget, travel_dates

# Main Page Sections
def render_home_page():
    st.markdown("""
    <div class="home-gradient">
        <h1>Welcome to Tiri Ya Tiyara!</h1>
        <p>Your AI-powered travel companion for unforgettable journeys</p>
    </div>
    """, unsafe_allow_html=True)

    # Action Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stCard">
            <h3>Popular Destinations</h3>
            <p>Explore trending travel spots</p>
            <button class="stButton">Discover</button>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stCard">
            <h3>Create Itinerary</h3>
            <p>Design your perfect trip</p>
            <button class="stButton">Plan Now</button>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stCard">
            <h3>Travel Tips</h3>
            <p>Expert advice and insights</p>
            <button class="stButton">Learn More</button>
        </div>
        """, unsafe_allow_html=True)

def render_plan_trip_page():
    st.title("Plan Your Dream Trip")
    
    with st.form("trip_planner"):
        destination = st.text_input("Where do you want to go?")
        duration = st.number_input("Trip Duration (Days)", min_value=1, max_value=30, value=7)
        interests = st.multiselect("Select Interests", 
            ["History", "Nature", "Food", "Adventure", "Culture", "Relaxation"])
        
        submitted = st.form_submit_button("Generate Itinerary")
        
        if submitted:
            st.success(f"Generating an amazing {duration}-day trip to {destination}!")

def render_saved_trips_page():
    st.title("My Saved Trips")
    st.markdown("""
    <div class="stCard">
        <p>You haven't saved any trips yet. Start planning your next adventure!</p>
        <button class="stButton">Create First Trip</button>
    </div>
    """, unsafe_allow_html=True)

def render_about_page():
    st.title("About Tiri Ya Tiyara")
    st.markdown("""
    <div class="home-gradient">
        <h2>Your Journey, Our Passion</h2>
        <p>We leverage cutting-edge AI to transform travel planning into an exciting, personalized experience.</p>
    </div>
    """, unsafe_allow_html=True)

def render_chat_interface():
    st.title("Chat with Tiri")
    
    # Initialize chat history in session state if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize chat instance if not exists
    if "chat_instance" not in st.session_state:
        st.session_state.chat_instance = GeminiChat()
    
    # Language selection
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox(
            "Input Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            key="source_lang"
        )
    with col2:
        target_lang = st.selectbox(
            "Response Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            key="target_lang"
        )
    
    # Chat input
    message = st.text_input("Type your message:", key="message_input")
    
    if st.button("Send"):
        if message:
            # Add user message to chat history
            st.session_state.chat_history.append(("user", message))
            
            responses = st.session_state.chat_instance.send_message(
                message, source_lang, target_lang
            )
            
            # Add responses to chat history
            for response in responses:
                if isinstance(response, AIMessage):
                    st.session_state.chat_history.append(("assistant", response.content))
    
    # Display chat history
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Tiri:** {content}")

def main():
    apply_custom_styling()
    page_selection, travel_type, budget, travel_dates = configure_sidebar()

    # Add Chat interface to navigation
    if page_selection == "Home":
        render_home_page()
    elif page_selection == "Plan My Trip":
        render_plan_trip_page()
    elif page_selection == "Saved Trips":
        render_saved_trips_page()
    elif page_selection == "About Us":
        render_about_page()
    
    # Add chat interface below main content
    st.markdown("---")
    render_chat_interface()

if __name__ == "__main__":
    main()