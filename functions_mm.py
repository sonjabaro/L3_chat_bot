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
    

#Define function to transcribes audio to text using Whisper in the original language it was spoken
def transcribe_audio_original(audio_filepath):
    try:
        transcription_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large")
        transcription_result = transcription_pipeline(audio_filepath)
        transcribed_text = transcription_result['text']
        return transcribed_text
    except Exception as e:
        print(f"an error occured: {e}")
        return "Error in transcription"
    


#Define function to transcribe audio to text and then translate it into the specified language
def translate(transcribed_text, target_lang="es"):
    try:
        #Define the model and tokenizer
        src_lang = detect(transcribed_text)
        model_name =f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        #tokenize the text
        encoded_text = tokenizer(transcribed_text, return_tensors="pt", padding=True)
        
        #generate translation using the model
        translated_tokens = model.generate(**encoded_text)
        
        #decode the translated tokens
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        return translated_text
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error in transcription or translation"


# Define function to translate text to speech for output
# Using Google Text-to-speech

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    return temp_file.name

#Define voice map
voice_map = {
    "ar": "Hala",
    "en": "Gregory",
    "es": "Mia",
    "fr": "Liam",
    "de": "Vicki",
    "it": "Bianca",
    "zh": "Hiujin",
    "hi": "Kajal",
    "jap": "Tomoko",
    "trk": "Burcu"
    
    }

#Define language map from full names to ISO codes
language_map = {
    "Arabic (Gulf)": "ar",
    "Chinese (Cantonese)": "zh",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "jap",
    "Spanish": "es",
    "Turkish": "trk"
    
}

# Define text-to-speech function using Amazon Polly
# Include voice map to specify which language requires which
# voice IDs

def polly_text_to_speech(text, lang_code):
    
    try:
    
        #get the appropriate voice ID from the mapping
        voice_id = voice_map[lang_code]
        
        #initialize boto3 client for polly
        polly_client = boto3.client('polly')
        
        #request speech synthesis
        response = polly_client.synthesize_speech(
            Engine = 'neural',
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice_id
        )
        
        # Save the audio to a temporary file and return its path
        if "AudioStream" in response:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as audio_file:
                audio_file.write(response['AudioStream'].read())
                return audio_file.name
    except boto3.exceptions.Boto3Error as e:
        print(f"Error accessing Polly: {e}")
    return None  # Return None if there was an error