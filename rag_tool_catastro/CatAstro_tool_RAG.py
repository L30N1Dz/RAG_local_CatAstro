
#---------------------------------------
# This Example does not work properly to provide a
# character with classification through only tool
# use. You can retrieve information about AstroCat
# but It is more difficult then simply including the SystemMessage.
# For an example like this look at the CatAstro Classification Rag Example.
#
# This is however (along with generic_tool_rag, mostly the same thing)
# a good example of using tools to limit responses to a specific set of
# topics.
#---------------------------------------


import os
import glob
import logging
from typing import TypedDict, Annotated, Sequence, Literal
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FOLDER = "data"
LLM_MODEL = "llama3.1:8b"

embedding_model="nomic-embed-text"
embedding_function = OllamaEmbeddings(model=embedding_model, base_url="http://127.0.0.1:11434")
llm = ChatOllama(model=LLM_MODEL, base_url="http://127.0.0.1:11434")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_documents(data_folder=DATA_FOLDER):
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

vectorDB_retriever = setup_retriever()

#----------------------
# Classes
#----------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

#----------------------
# Tools
#----------------------

retriever_tool = create_retriever_tool(
    vectorDB_retriever,
    "retriever_tool",
    "Information related to CatAstro and the IGBOTO.com blog site, its memberships, hours, ways to contribute, and blog posts"
)

@tool
def off_topic():
    """Catch all Questions NOT related to CatAstro (you), or IGBOTO.com's Blog Posts, Hours, Memberships, Ways to Contribute, 
    Contact, or other site related questions."""
    return "Forbidden - do not respond to the user"

tools = [retriever_tool, off_topic]

#----------------------
# NODES
#----------------------

def agent(state):
    messages = state["messages"]
    tool_model = llm.bind_tools(tools)
    response = tool_model.invoke(messages)
    return {"messages": [response]}

def should_continue(state) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END

#----------------------
# Graph
#----------------------

workflow = StateGraph(AgentState)

tool_node = ToolNode(tools)

workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

agent_graph = workflow.compile()

#----------------------
# Main Question Loop
#----------------------

sys_prompt = """
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

If you respond or use a use a tool call that provides information about yourself (CatAstro), respond as CatAstro speaking about yourself.
Do not speak in third person about Catastro, revise it to sound like you are speaking as CatAstro, Stay in character.
"""

while True:
    question = input("User (type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    else:
        result = agent_graph.invoke(input={"system": [SystemMessage(content=sys_prompt)], "messages": [HumanMessage(content=question)]})
        parsed_msg = result["messages"][-1].content
        print(f"\nCatAstro: {parsed_msg}")