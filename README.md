# RAG_local_CatAstro
This is a simple example of using RAG locally (Ollama) w/ CatAstro Example Data.

Description:
This is a simple example of how to perform RAG, Retrieval Augmented Generation and integrate it with a Large Language Model
running on a local Ollama instance.

The rag_simple_catastro example showcases a single file script that loads txt files from the data folder.
The data is vectorized using nomic-embed-text and stored in a db in memory.
The retriever uses MMR (Maximum Marginal Relavance) and retrieves 3 chunks.
We then feed our question and relevant chunks to the LLM using a custom prompt.

This example data is for use by a web chat assistant to answer questions about a website.
For fun I've added a system prompt to give the AI a character, I've included some AI generated adventures,
and a bio to give the chatbot a more interesting and personalized touch for the site.

This is an example to show off the simplicity of Local RAG with Ollama, it runs in a terminal and should only be 
used as an example to buiild upon.

To run these examples you will need to install Ollama and at least two models:
1) llama3.1:8b (language model used in most of the examples, you can change this in the code if you would like)
2) nomic-embed-text (the Vectorizing model used to generate the DB)
3) *optional* mxbai-embed-large (also a text vectorizing model, more advanced)

There are plenty of Guides on installing Ollama but a quick refresher on basic terminal commands:

ollama list (view what models are installed)

ollama pull [model_name] (download a model)

As long as Ollama is running you do not have to start the model unless you want it to continue running in the backround,
when the script makes a call to Ollama it will start the  proper model, run the prompts and unload automatically.

Prerequisits: (requirements.txt, Please install these packages to run the script.)

langchain

langchain-community

langchain-core

langchain-ollama

chromadb ( For some reason pipreqs did not pick this one up for requirements.txt but you will likely need to install it )

  *NOTES:* 

For the sample data, a lot of it was generated or modified by AI, although it was simple text and the files said they where UTF-8 encoded, they would not load properly.
In my case there was an offending 0x9D character in the text files which does not conform to UTF-8 encoding. I used a Hexeditor (HXD) to find and replace the 0x9D values with 0x0D? (decimal place)
Or I could have simply remove them. If your txt files do not load properly check this, I will upload the script I used to find the offending value if anyone else has this issue.

![alt text](https://github.com/L30N1Dz/RAG_local_CatAstro/blob/main/CatAstro's_Space_Adventures.png?raw=true)
