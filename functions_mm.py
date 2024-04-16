from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM
import subprocess
import torch
#Google Text to Speech
from gtts import gTTS
import tempfile
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import boto3
# Additional imports for loading PDF documents and QA chain.
from langchain_community.document_loaders import PyPDFLoader
# Additional imports for loading Wikipedia content and QA chain
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#Setting the Model


# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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


#Define combined function to feed into Gradio app
def combined_function (audio_filepath, target_lang):
    
    #language detection for the original text
    transcribed_text = transcribe_audio_original(audio_filepath)
    detected_lang = detect(transcribed_text)
    
    # query the diabetes model with the transcibed text
    response_text = handle_query(transcribed_text)
    
    
    # translate the response text to the target language
    target_lang_code = language_map[target_lang]
    translated_response = translate(response_text, target_lang_code)
    
    #text to speech for original and translated response 
    original_speech = polly_text_to_speech(response_text, detected_lang)
    translated_speech = polly_text_to_speech(translated_response, target_lang_code)
    
    
    return transcribed_text, response_text, translated_response, original_speech, translated_speech

#########################################################################

def transcribe_and_speech(audio_filepath=None, typed_text=None, target_lang=default_language):
    
    #Determine source of text: audio transctiption or direct text input
    if audio_filepath and typed_text:
        return "Please use only one input method at a time", None
    
    query_text = None
    detected_lang_code = None
    original_speech = None
    
    if typed_text:
        #convert typed text to speech
        query_text = typed_text
        detected_lang_code = detect(query_text)
        original_speech = polly_text_to_speech(query_text, detected_lang_code)
        return None, original_speech
    
    elif audio_filepath:
        #transcribe audio to text
        query_text = transcribe_audio_original(audio_filepath)
        detected_lang_code = detect(query_text)
        original_speech = polly_text_to_speech(query_text, detected_lang_code)
        return query_text, original_speech
    
    if not query_text:
        return "Please provide input by typing or speaking.", None
    
    #Check if the language is specified. Default to English if not.
    target_lang_code = language_map.get(target_lang, "en")
    
    #Map detected language code to language name
    detected_lang = [key for key, value in language_map.items() if value == detected_lang_code][0]
    
    
    return query_text, original_speech







































default_language = "English"

def translate_and_speech(query_text=None, typed_text=None, target_lang=default_language):
    
    #Determine source of input: transcribed text from audio filepath or direct text input
    
    if query_text and typed_text:
        return "Translate button will translate the transcribed text box or the text input box, but not both concurrently. Please ensure only one box is populated.", None
    elif typed_text:
        to_translate_text = typed_text
    elif query_text: 
        to_translate_text = query_text
    else: "Please provide input by typing ", None, None, None
        
    #Detect language of input text
    detected_lang_code = detect(to_translate_text)
    detected_lang = [key for key, value in language_map.items() if value == detected_lang_code][0]
    
    #Check if the language is specified. Default to English if not.
    target_lang_code = language_map.get(target_lang, "en")
    
    #Process text: translate 
    #Check if the detected language and target language are the same
    if detected_lang == target_lang:
        translated_text = to_translate_text
    else:
        translated_text = translate(to_translate_text, target_lang_code)
    
    #convert to speech
    translated_speech = polly_text_to_speech(translated_text, target_lang_code)
    
    return  translated_text, translated_speech