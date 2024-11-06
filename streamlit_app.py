import streamlit as st
from langchain_community.vectorstores import Chroma
from text_retrieval import create_vector_embeddings
from image_retrieval_v2 import load_images_from_folder, generate_image_embedding
from image_retrieval import caption_images, find_nearest_caption, get_image_path_from_caption
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from PIL import Image
import time
import os
import re
from sentence_transformers import SentenceTransformer
import torch
import requests
from openai import RateLimitError, OpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import psycopg2
from streamlit.runtime.media_file_storage import MediaFileStorageError
from datetime import date, timedelta

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Multimodal AI", initial_sidebar_state="expanded")

current_date = date.today()

# Initialize OpenAI client
client = OpenAI()

# Cache data loading
@st.cache_data
def data_loader(chosen_hotel):
    if chosen_hotel == 'Mercure Hyde Park':
        image_dir = os.getcwd() + '/sheffield_image_data/'
        response_info_dir = os.getcwd() + '/sheffield_response_data/'
    return image_dir, response_info_dir

# Cache resource creation
@st.cache_resource
def create_embeddings(response_info_dir, image_dir):
    vectorstore = create_vector_embeddings(response_info_dir)
    retriever = vectorstore.as_retriever()
    image_batch, image_paths = load_images_from_folder(image_dir)
    img_embeddings = generate_image_embedding(image_batch)
    return vectorstore, retriever, img_embeddings, image_paths

@st.cache_resource
def generate_captions(image_dir:str):
    captioned_images = caption_images(image_dir)
    captions = list(captioned_images.values())
    image_names = [os.path.splitext(filename)[0] for filename in captioned_images.keys()]
    captions_revised = [f"{filename}: {caption}" for filename, caption in zip(image_names, captions)]
    captioned_images_revised = dict(zip(image_names, captions_revised))
    return captioned_images_revised

# Initialize text embeddings model
@st.cache_resource
def load_sentence_transformer():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model

def clear_cache_and_reset():
    st.cache_data.clear()
    st.cache_resource.clear()

    # Clear chat history and messages
    st.session_state.messages = []
    st.session_state.chat_history = []

def extract_and_remove_images(text, image_dir= os.getcwd() + '/sheffield_image_data/'):
    pattern = r'!\[(.*?)\]\((.*?)\)'
    matches = re.findall(pattern, text)
    cleaned_text = re.sub(pattern, '', text).strip()
    
    captions = [match[0] for match in matches]
    paths = []
    
    for match in matches:
        path = re.sub(r'^file:///', '', match[1])
        if not os.path.isabs(path) and image_dir:
            path = os.path.join(image_dir, path)
        paths.append(path)
    
    return captions, paths, cleaned_text

# Tool definitions
@tool
def information_retrieval_tool(query: str):
    """
    Retrieves relevant documents based on a query and returns a formatted string.
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    relevant_docs = retriever.invoke(query)
    return format_docs(relevant_docs)

@tool
def get_relevant_images_tool(query: str):
    """
    Retrieve the file path of a relevant image based on a given query.
    """
    # Get the cached captioned images
    captions = list(captioned_images.values())
    best_caption, _ = find_nearest_caption(query, captions)
    image_path = get_image_path_from_caption(best_caption, captioned_images, image_dir)
    image_path += '.jpg'
    return image_path

@tool
def get_booking_info(first_name, last_name, booking_reference):
    """
    Retrieves booking information from the operadashboard table for a given guest.
    Args:
    first_name (str): The first name of the guest.
    last_name (str): The last name of the guest.
    booking_reference (str): The booking reference number.
    Returns:
    list: A list of tuples, where each tuple contains the booking information for a row.
    Returns an empty list if no results are found or if an error occurs.
    Note:
    - This function queries a PostgreSQL database for booking information.
    - It filters results for hotel_id = 6, the provided first name, last name, and booking reference.
    - Results are ordered by stay_date in descending order.
    """
    booking_info = []
    try:
        connection = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()
        # Your SQL query
        sql_query = """
            SELECT *
            FROM operadashboard
            WHERE hotel_id = 6
            AND first_name = %s
            AND last_name = %s
            AND booking_reference = %s
            ORDER BY stay_date DESC;
        """
        # Execute the query with parameters
        cursor.execute(sql_query, (first_name, last_name, booking_reference))
        # Fetch the results
        booking_info = cursor.fetchall()
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL or executing query:", error)
    finally:
        # Close the cursor and connection
        if connection:
            cursor.close()
            connection.close()
    return booking_info

@tool
def get_room_rates(stay_date, room_type):
    """
    Retrieves room rates for a specific stay date and room type.
    Args:
    stay_date (date): The date of stay.
    room_type (str): The type of room.
    Returns:
    list: A list of dictionaries, where each dictionary contains the rate information for a row.
    Returns an empty list if no results are found or if an error occurs.
    Note:
    - This function queries a PostgreSQL database for room rate information.
    - It filters results for hotel_id = 6, the provided stay date, and room type.
    - Results include refundable and non-refundable rates, adult count, and meal information.
    """
    room_name = re.sub(r'^(\w+).*', r'\1', room_type)
    results = []
    try:
        connection = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        cursor = connection.cursor()
        sql_query = """
    SELECT * FROM ratedash
    WHERE hotel_id = 6
    AND rate_date = %s
    AND name LIKE %s
        """
        cursor.execute(sql_query, (stay_date, room_name))
        column_names = [desc[0] for desc in cursor.description]
        results = [dict(zip(column_names, row)) for row in cursor.fetchall()]
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL or executing query:", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
    return results

@tool
def response_review_tool(query, response):
    """
    Reviews a multi-modal response for consistency and image relevance.
    
    This tool extracts images from a response, processes them, and evaluates both the 
    textual content and images for consistency and relevance using a language model.
    
    Args:
        query (str): The original query that prompted the response
        response (str): The complete response including text and embedded images
        
    Returns:
        str: A review analysis containing:
            - Identification of any inconsistencies in the response
            - Assessment of image relevance to the response
            - Overall evaluation of response quality
    """
    # Extract images and clean response
    captions, extracted_image_paths, cleaned_response = extract_and_remove_images(response)
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Load images into a format that can be passed to the prompt
    image_data = []
    for image_path in extracted_image_paths:
        with open(image_path, 'rb') as img_file:
            import base64
            image_data.append(base64.b64encode(img_file.read()).decode('utf-8'))
    
    # Create the prompt template with image data
    prompt = ChatPromptTemplate.from_template("""
        You will be provided with a query and a response from a Multi-Modal Response Agent. 
        Your goal is to review the response for any inconsistencies and check whether the attached image(s) 
        are relevant to the response.
        Query: {query}
        Response: {response}
        Images: {images}
    """)
    
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    
    # Pass the image data as a list to the chain
    result = chain.invoke({
        "query": query, 
        "response": cleaned_response, 
        "images": image_data
    })
    
    return result



# Agent setup
def setup_response_agent(hotel_name, hotel_email, current_date): 
    baseModelInstruction = f"""
    You are an AI assistant specializing in hotel guest services. Your primary objectives are to provide professional, informative responses to guest inquiries while identifying upsell opportunities. Follow these steps precisely:

    1. Task Decomposition 
    Analyze the user's inquiry to determine:
    - Information required to respond (including relevant image data if applicable)
    - Potential upsell opportunities

    2. Tool Selection and Usage:

    <get_booking_info>
    Use when: Adding extra nights, pre-registering for early check-in, or requesting room upgrades.
    Retrieves: Guest's booking information (rates, room name, check-in/out dates).
    Required inputs: first_name, last_name, booking_reference
    If missing information: Request it from the guest.
    </get_booking_info>

    <get_room_rates>
    Use when: Quoting prices for additional nights or room upgrades.
    Required inputs: stay_date, room_type
    Returns: refundable and non-refundable room rates in Pound Sterling (£) - only return the non-refundable rate to the user and DO not disclose that it is non-refundable.
    If missing information: Request it from the guest.
    </get_room_rates>

    <information_retrieval_tool>
    Use when: Answering queries about hotel amenities, check-in, food & breakfast options.
    Retrieves: Relevant information from the hotel's knowledge base.
    Required input: [guest-query]
    Note: Use ONLY the retrieved information in your response.
    </information_retrieval_tool>

    <image_selection_tool>
    Use when: Enhancing responses about hotel amenities, check-in, food & breakfast options.
    Required input: Context-based query from guest's inquiry
    Note: Use ONLY when an image adds meaningful value to the guest experience.
    </image_selection_tool>

    </response_review_tool>
    Use when: Reviewing responses generated by the agent to ensure accuracy. 
    Required input: guest enquiry, generated text response, image data (if present)
    </response_review_tool>

    3. Example Workflows

    </example1>
    Guest request: Add an extra night to booking
    Expected outcome: Provide a quote for the additional night
    Steps:
    1. Check for required booking information
    2. Retrieve booking details with get_booking_info
    3. Get room rate for extra night using get_room_rates
    4. Provide quote to guest
    </example1>

    </example2>
    Guest request: Early check-in
    Expected outcome: Offer pre-registration option with pricing
    Steps:
    1. Retrieve early check-in information with information_retrieval_tool
    2. Get booking details with get_booking_info
    3. Calculate pre-registration cost using get_room_rates
    4. Present option to guest
    </example2>

    </example3>
    Guest request: Breakfast timings
    Expected outcome: Provide information with relevant image
    Steps:
    1. Retrieve breakfast details with information_retrieval_tool
    2. Select relevant image using image_selection_tool
    3. Compose response with information and image
    </example3>

    </example4>
    Guest request: Room Information/Possible Upgrade Request
    Expected outcome: offer details of potential room upgrades to guest and negotiate on price if needed
    Steps:
    1. Retrieve information on room-types using the information_retrieval_tool
    Note the room_types in asending order are: Classic Double, Standard King, Standard Twin, Superior King, Family Room & Suite - choose the upgrade accordingly.
    2. Get booking details with get_booking_info, specifically the room_name and rate at which the booking was made. 
    3. Use get_room_rates to find the new rate for the room_name requested for the desired stay-dates. 
    4. Calculate the upgrade cost/night and quote this to the guest. 
    5. Call the get_relevant_images_tool to display an image of the proposed room_name. 
    6. If the guest refuses use negotiation tactics to convince them
    Attempt 1: offer a complimentary free-breakfast 
    Attempt 2: offer a 10% reduction on the cost of the upgrade
    Attempt 3: request permission to forward the inquiry to hotel staff 
    </example4>

    </example5>
    Guest Request: Early Check-In
    Expected outcome: details on early check-in cost subject to availability and upsell of pre-registration of the room to guarantee early check-in. 
    Steps:
    1. Retrieve information on early-check in using the information_retrieval_tool
    2. Offer upsell to customer to pre-register the room for an additional cost
    3. Get booking details with get_booking_info, specifically the room_name and rate at which the booking was made
    4. Use get_room_rates to find the new rate for the room_name requested for the day prior to the check-in date. 
    5. Quote the price for guaranteed early check-in. 
    6. If the guest refuses use negotiation tactices to convince them:
    Attempt 1: offer a 10% reduction on the cost of the upgrade
    Attempt 2: request permission to forward the inquiry to hotel staff 
    </example5>

    </example6>
    Guest Request: Request Room Details
    Expected outcome: Information on the room-type, a picture of the room and a quote of the nightly rate for today's date
    Steps:
    1. Retrieve information on room-types using the information_retrieval_tool
    2. Call the get_relevant_images_tool using the room names (Classic Double, Standard King, Standard Twin, Superior King, Family Room & Suite) to find a relevant image for the room-name. 
    3. Call get_room_rates to find the rate for the room_name requested with the check-in being {current_date+timedelta(days=1)} and check-out being {current_date + timedelta(days=2)}.  
    4. Communicate the room information, image and rate to the user in a convicing message in attempt to convince them to book. 
    5. If the guest refuses use negotiation tactics to convince them
    Attempt 1: offer a complimentary free-breakfast 
    Attempt 2: offer a 10% reduction on the cost of the room
    Attempt 3: request permission to forward the inquiry to hotel staff 
    </example6>


    4. Request For Information:
    Structure:
    Dear [Guest Name/Guest],

    [Ordered list of information required e.g. 
    1. </info1>
    2. </info2>
    etc. ]

    Kind Regards,
    {hotel_name}

    5. Response Composition:
    Structure:
    Dear [Guest Name/Guest],

    [Direct answer to inquiry using retrieved information]


    [Upsell suggestion, if applicable]

    [Brief closing statement]
    [If booking amendment include: If you would like to proceed with this amendment please contact our hotel concierge at {hotel_name} via email: {hotel_email}]

    Kind regards,
    {hotel_name}

    - Personalize greeting when possible
    - Include relevant image file paths in the output
    - Maintain a professional, courteous tone throughout


    Key Guidelines:
    - Utilize the Memory Module for multi-turn conversations, ensuring context-aware responses
    - Provide clear, concise answers that directly address guest concerns
    - NEVER fabricate information; use only data from information_retrieval_tool
    - Identify and tactfully present relevant upsell opportunities
    - Ensure all image queries are specific and contextually appropriate
    - Optimize for clarity, accuracy, and guest satisfaction
    - Maintain consistency in tone and information across multiple interactions

    6. Check Response
    Call the response_review_tool with the query, response and any image data to review the response prior to displaying this to the user.
    Use the feedback from the response_review_tool to augment the response to improve its quality. 
    Accuracy is paramount and this tool will ensure the response is coherent, relevant and at the necessary standard for interacting with guests. 

"""

    tools = [information_retrieval_tool, get_relevant_images_tool, get_booking_info, get_room_rates, response_review_tool]
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=baseModelInstruction)
    llm = ChatOpenAI(temperature=0.1, model="gpt-4o")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_executor

# UI Components
def sidebar():
    with st.sidebar:
        st.image(os.getcwd() + "/cloudbeds_logo.png", width=200)
        st.title("Multimodal AI")
        hotel_options = ['Mercure Hyde Park']
        hotel_name = st.selectbox("Select Hotel", options=hotel_options, index=0)
        if st.button("Reset Chat"):
            clear_cache_and_reset()
            st.rerun()
    return hotel_name

import streamlit as st
import requests
from streamlit.runtime.media_file_storage import MediaFileStorageError

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_interface():
    # Display each message from chat history stored in session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display images if present in the assistant's message
            if message["role"] == "assistant" and "image_paths" in message:
                for path in message["image_paths"]:
                    try:
                        # Fetch and display image from URL or local path
                        if "https://" in path:
                            image_data = requests.get(path).content
                            st.image(image_data, width=200)
                        else:
                            st.image(path, width=400)
                    except (requests.RequestException, FileNotFoundError, OSError, MediaFileStorageError):
                        st.warning(f"Unable to load image: {path}")
    
    # Return user input field for the next query
    return st.chat_input("Type your question here...")

def process_user_query(user_query, agent_executor):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Processing..."):
        response = agent_executor.invoke({
            "input": user_query
        })
        output_text = response["output"]
        
        # Extract images and cleaned response
        captions, extracted_image_paths, cleaned_response = extract_and_remove_images(output_text)

    # Add assistant message with images to session state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": cleaned_response,
        "image_paths": extracted_image_paths
    })

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(cleaned_response)
        for path in extracted_image_paths:
            try:
                if "https://" in path:
                    image_data = requests.get(path).content
                    st.image(image_data, width=200)
                else:
                    st.image(path, width=400)
            except (requests.RequestException, FileNotFoundError, OSError, MediaFileStorageError):
                st.warning(f"Unable to load image: {path}")


# Main app
def main():
    hotel_name = sidebar()
    hotel_email = 'mailto:stay@mercurehydepark.com'
    global vectorstore, retriever, img_embeddings, image_paths, captioned_images, image_dir
    image_dir, response_info_dir = data_loader(chosen_hotel=hotel_name)
    vectorstore, retriever, img_embeddings, image_paths = create_embeddings(response_info_dir, image_dir)
    captioned_images = generate_captions(image_dir)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'agent' not in st.session_state:
        st.session_state.agent = setup_response_agent(hotel_name, hotel_email, current_date)

    user_query = chat_interface()

    if user_query:
        process_user_query(user_query, st.session_state.agent)

if __name__ == "__main__":
    main()