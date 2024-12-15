# Import required modules
import asyncio
import streamlit as st
from langchain_chroma import Chroma  # Vector database
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Google's LLM and embeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from langgraph.prebuilt import create_react_agent  # For creating an agent that can use tools
from langchain.tools import tool  # Decorator to create tools
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


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
        
    def send_message(self, message: str) -> str:
        """
        Send a message and get a response from the model.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response content
        """
        # Add user message to history
        self.messages.append(HumanMessage(content=message))
        
        # Store current history length to identify new messages later
        history_length = len(self.messages)
        
        # Get response from agent, including any tool usage
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        
        # Extract only the new messages from this interaction
        new_messages = self.messages[history_length:]

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

def main():
    apply_custom_styling()
    page_selection, travel_type, budget, travel_dates = configure_sidebar()

    # Page Routing
    if page_selection == "Home":
        render_home_page()
    elif page_selection == "Plan My Trip":
        render_plan_trip_page()
    elif page_selection == "Saved Trips":
        render_saved_trips_page()
    elif page_selection == "About Us":
        render_about_page()

if __name__ == "__main__":
    main()