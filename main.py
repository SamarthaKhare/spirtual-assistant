from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel,GenerationConfig
import os
from fastapi.middleware.cors import CORSMiddleware
from sys_pr import response_schema,system_prompt

app = FastAPI()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'emerald-bastion-433109-e0-fb7a9d5222ab.json'
PROJECT_ID = "zif5x-437508"
FILE_PATH="https://drive.google.com/file/d/1foSCX4VkNneL-b23nXik4bTNXoEobTOQ"
DISPLAY_NAME = "trial_corpus"
vertexai.init(project=PROJECT_ID, location="us-central1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

def generate_response(question: str):
    try:
        existing_corpora = rag.list_corpora()
        corpus = next((c for c in existing_corpora if c.display_name == DISPLAY_NAME), None)
        if not corpus:
            embedding_model_config = rag.EmbeddingModelConfig(
                publisher_model="publishers/google/models/text-embedding-004"
            )
            backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)
            corpus = rag.create_corpus(display_name=DISPLAY_NAME, backend_config=backend_config)
            print("RAG Corpus Created:", corpus.name)

        corpus_name = corpus.name
        files = rag.list_files(corpus_name)
        if not any(file.display_name == "book.pdf" for file in files):
            print('no file in corpus yet')
            transformation_config = rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=200),
            )
            rag.import_files(corpus_name=corpus_name, paths=[FILE_PATH], transformation_config=transformation_config)
            print("JSONL file imported into RAG corpus with per-document embeddings.")
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=5,
            filter=rag.Filter(
                vector_distance_threshold=0.5, 
            )
        )
        # Get the relevant context from book
        retrieval_response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            text=question,
            rag_retrieval_config=rag_retrieval_config,
        )
        #retrived contexts
        context=retrieval_response.contexts.contexts
        rag_model = GenerativeModel(model_name="gemini-2.0-flash",system_instruction=system_prompt)
        prompt = f"""
            Below is the user question stating his moral or spirtual confusion: \n
            {question}
            Below is the relevant context from Bhagavad Gita understand it and use it answer the question: \n
            {context}
        """
        generation_config = GenerationConfig(
            temperature=0.3,
            top_p=1,
            top_k=2,
            max_output_tokens=8000,
            response_mime_type="application/json",
            response_schema=response_schema
        )
        print('answer prepared by the model')
        response = rag_model.generate_content(prompt, generation_config=generation_config)
        response_json = json.loads(response.text)
        return response_json
    
    except Exception as e:
        error_data = {
            "error": str(e)
        }
        with open("error_log.json", "w") as f: 
            json.dump(error_data, f)
            f.write("\n")  
        print('error detected')
        return {}


@app.post("/chat")
def chat(request: ChatRequest):
    response = generate_response(request.query)
    if response:
        return {"answer": response}
    else:
        raise HTTPException(status_code=500, detail="Error generating response")

# Run the server (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
