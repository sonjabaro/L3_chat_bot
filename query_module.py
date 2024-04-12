from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.schema import AIMessage, HumanMessage
import gradio as gr

load_dotenv()

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Additional imports for loading PDF documents and QA chain.
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# Additional imports for loading Wikipedia content and QA chain.
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains.question_answering import load_qa_chain

#Setting the Model

# Initialize the model with your OpenAI API key
load_dotenv()

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = 'gpt-3.5-turbo'  # We're using GPT 3.5 turbo as we don't really need GPT 4 for this as the info doesn't change that much. It provides what we need at a lower cost.
#Instantiating the llm we'll use and the arguments to pass
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0)

# Define the wikipedia topic as a string.
def initialize_model():
    global documents, chain
    #load documents
    wiki_topic = "diabetes"

    # Load the wikipedia results as documents, using a max of 2.
    #included error handling- unable to load documents
    try:
        documents = WikipediaLoader(query=wiki_topic, load_max_docs=2, load_all_available_meta=True).load()
    except Exception as e:
        print("Failed to load documents:", str(e))
        documents = []
    
    # Create the QA chain using the LLM.
    chain = load_qa_chain(llm)

# Call the initialization at the start
initialize_model()

#Define the function to call the LLM
def handle_query(user_query):
    global documents, chain
    if not documents:
        return "Source not loading info; please try again later."

    if user_query.lower() == 'quit':
        return "Goodbye!"

    try:
        # Pass the documents and the user's query to the chain, and return the result.
        result = chain.invoke({"input_documents": documents, "question": user_query})
        return result["output_text"] if result["output_text"].strip() else "No answer found, try a different question."
        
    except Exception as e:
        return "An error occurred while searching for the answer: " + str(e)