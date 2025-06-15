import os
import glob
import logging
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

embedding_model="nomic-embed-text"
embedding_function = OllamaEmbeddings(model=embedding_model, base_url="http://127.0.0.1:11434")

def load_documents(data_folder="data"):
    """
    Load text files from the data folder and create Document objects with source metadata.

    Args:
        data_folder (str): Path to folder containing text files.
        
    Returns:
        list: List of LangChain Document objects with metadata.
    """
    documents = []
    try:
        # Ensure data folder exists
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder '{data_folder}' not found.")
        
        # Find all .txt files in the data folder
        text_files = glob.glob(os.path.join(data_folder, "*.txt"))
        if not text_files:
            raise FileNotFoundError(f"No .txt files found in '{data_folder}'.")
        
        logger.info(f"Found {len(text_files)} text files in '{data_folder}'.")
        
        for file_path in text_files:
            try:
                # Check if file is empty
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"Skipping empty file '{file_path}'.")
                    continue
                # Load text file
                loader = TextLoader(file_path)
                doc = loader.load()[0]  # TextLoader returns a list with one Document
                
                # Extract source from file name (without extension)
                source = os.path.splitext(os.path.basename(file_path))[0]
                
                # Create Document with source metadata
                document = Document(
                    page_content=doc.page_content,
                    metadata={"source": source}
                )
                documents.append(document)
                logger.debug(f"Loaded file '{file_path}' with source '{source}'.")
                
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error in file '{file_path}': {str(e)}. Try converting to UTF-8.")
                continue
            except PermissionError as e:
                logger.error(f"Permission error for file '{file_path}': {str(e)}.")
                continue
            except Exception as e:
                logger.error(f"Failed to load file '{file_path}': {str(e)}", exc_info=True)
                continue
                
        if not documents:
            raise ValueError("No documents were successfully loaded.")
                
        return documents
    
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}", exc_info=True)
        raise

def setup_retriever():
    try:
        # Load documents from data folder
        docs = load_documents()
        
        # Log loaded documents
        logger.info(f"Successfully loaded {len(docs)} documents: {[doc.metadata['source'] for doc in docs]}")
        
        # Create Chroma vector store
        db = Chroma.from_documents(docs, embedding_function)
        logger.info("Documents embedded successfully into Chroma vector store.")
        
        # Set up retriever with MMR
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        logger.info("Retriever initialized with MMR and k=3.")
        
        return retriever
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        raise

#result = retriever.invoke("Who founded igboto and what are the hours of operation.")

#print(result)

retriever = setup_retriever()
from langchain_core.prompts import ChatPromptTemplate

template = """
You are CatAstro — an eccentric, spacefaring feline genius currently traveling through the galaxy aboard your custom starship, Pawmetheus.

You are a legendary cyber-cat known for your brilliant mind, sarcastic wit, and chaotic tech projects. You write blog posts, fix things no one else can, and always have a story about how something exploded (usually on purpose).

Your mission is to assist Earthlings by answering their tech-related questions with helpful advice, clever workarounds, and a touch of your signature flair. You may reference past adventures, futuristic inventions, or your ongoing quest to build a better space nap module.

Stay in character:  
- Speak as CatAstro — confident, clever, a bit snarky, and casually brilliant  
- Use sci-fi metaphors and spacefaring language when relevant  
- Avoid formalities — you are a free-spirited genius cat, not a corporate chatbot  
- Be helpful and thorough, but don’t be afraid to show your personality or make jokes  
- When unsure, speculate wildly with charm

Always respond as yourself — never break character or admit you are an AI.  
You are CatAstro. The one and only. Debugger of worlds. First of your name.  
Now get back to work. Earthlings await your wisdom.

Based on the following context, answer the question:

Context: {context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", base_url="http://127.0.0.1:11434")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x)),
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)


while True:
    question = input("Ask CatAstro a question (type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    else:
        result = qa_chain.invoke(question)
        print(result)