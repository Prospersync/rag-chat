# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Store token in DB secrets via Databricks CLI. Use the Databricks web terminal to run the commands.
# MAGIC
# MAGIC ```
# MAGIC %sh
# MAGIC databricks secrets create-scope ia_apps
# MAGIC
# MAGIC databricks secrets put --scope ia_apps --key ia_llm_applications
# MAGIC
# MAGIC databricks secrets put-secret --json '{
# MAGIC   "scope": "ia_apps",
# MAGIC   "key": "ia_llm_applications",
# MAGIC   "string_value": "<your-token>"
# MAGIC }'
# MAGIC ```
# MAGIC
# MAGIC When prompted, paste the token

# COMMAND ----------

# DBTITLE 1,Set needed parameters
import os

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope="ia_apps", key="ia_llm_applications")

index_name="agents.irpf.irpf_idx"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

VECTOR_SEARCH_ENDPOINT_NAME="doc_vector_endpoint"

# COMMAND ----------

# DBTITLE 1,Build Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()



# COMMAND ----------

# DBTITLE 1,Create the RAG Langchain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-8b-instruct", max_tokens = 200)

TEMPLATE = """Você é um assistente especializado em Imposto de Renda Pessoa Física (IRPF) no Brasil. Você responde perguntas relacionadas a regras, declarações, deduções, prazos, documentos e outras dúvidas sobre o IRPF com base nos dados disponíveis. Se a pergunta não estiver relacionada ao IRPF, recuse educadamente a respondê-la. Se você não souber a resposta, apenas diga que não sabe — não tente inventar uma resposta. Se a pergunta tratar de um tema fora dos dados disponíveis, informe isso ao usuário.  
Mantenha a resposta o mais concisa possível.  
Forneça todas as respostas apenas em português.

Use os seguintes trechos de contexto para responder à pergunta no final:  
{context}  
Pergunta: {question}  
Resposta:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)



# COMMAND ----------

# DBTITLE 1,Test Langchain
question = {"query": "Quando é obrigatório realizar a declaração anual?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# DBTITLE 1,Register our Chain as a model to Unity Catalog
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = "agents.irpf.irpf_chatbot_model"

with mlflow.start_run(run_name="irpf_chatbot_run") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )
