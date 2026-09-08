from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from src.config import settings
from src.llm import get_embeddings, get_llm
from src.utils.logging_config import wrap_vector_store_with_logging

# NOTE: When switching between OpenAI (1536d) and Ollama (768d for nomic-embed-text),
# the Neo4j 'reviews' vector index must be rebuilt to avoid embedding dimensionality errors.
neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=get_embeddings(),
    url=settings.NEO4J_URI,
    username=settings.NEO4J_USERNAME,
    password=settings.NEO4J_PASSWORD,
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)
neo4j_vector_index = wrap_vector_store_with_logging(neo4j_vector_index, "reviews")

review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=get_llm(model=settings.HOSPITAL_QA_MODEL, temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt
