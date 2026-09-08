from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from src.config import settings
from src.llm import get_embeddings, get_llm
from src.utils.logging_config import wrap_vector_store_with_logging
from src.langchain_custom.graph_qa.cypher import GraphCypherQAChain

graph = Neo4jGraph(
    url=settings.NEO4J_URI,
    username=settings.NEO4J_USERNAME,
    password=settings.NEO4J_PASSWORD,
)

graph.refresh_schema()

# NOTE: Vector dimensions must match between provider and Neo4j vector index
# (1536d for OpenAI text-embedding-ada-002 vs 768d for Ollama nomic-embed-text)
cypher_example_index = Neo4jVector.from_existing_graph(
    embedding=get_embeddings(),
    url=settings.NEO4J_URI,
    username=settings.NEO4J_USERNAME,
    password=settings.NEO4J_PASSWORD,
    index_name=settings.NEO4J_CYPHER_EXAMPLES_INDEX_NAME,
    node_label=settings.NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY.capitalize(),
    text_node_properties=[
        settings.NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY,
    ],
    text_node_property=settings.NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY,
    embedding_node_property="embedding",
)
cypher_example_index = wrap_vector_store_with_logging(
    cypher_example_index, settings.NEO4J_CYPHER_EXAMPLES_INDEX_NAME
)

cypher_example_retriever = cypher_example_index.as_retriever(search_kwargs={"k": 8})

cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly (e.g. [c:COVERED_BY] instead of [:COVERED_BY]). Do not run any
queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH v as visit, c.billing_amount as billing_amount)
If you need to divide numbers, make sure to
filter the denominator to be non zero.

Example queries for this schema:
{example_queries}

Warning:
- Never return a review node without explicitly returning all of the properties
besides the embedding property
- Make sure to use IS NULL or IS NOT NULL when analyzing missing properties.
- You must never include the
statement "GROUP BY" in your query.
- Make sure to alias all statements that
follow as with statement (e.g. WITH v as visit, c.billing_amount as
billing_amount)
- If you need to divide numbers, make sure to filter the denominator to be non
zero.

String category values:
Test results are one of: 'Inconclusive', 'Normal', 'Abnormal'
Visit statuses are one of: 'OPEN', 'DISCHARGED'
Admission Types are one of: 'Elective', 'Emergency', 'Urgent'
Payer names are one of: 'Cigna', 'Blue Cross', 'UnitedHealthcare', 'Medicare',
'Aetna'

If you're filtering on a string, make sure to lowercase the property and filter
value.

A visit is considered open if its status is 'OPEN' and the discharge date is
missing.

Use state abbreviations instead of their full name. For example, you should
change "Texas" to "TX", "Colorado" to "CO", "North Carolina" to "NC", and so on.

The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "example_queries", "question"],
    template=cypher_generation_template,
)

qa_generation_template = """You are an assistant that takes the results from
a Neo4j Cypher query and forms a human-readable response. The query results
section contains the results of a Cypher query that was generated based on a
user's natural language question. The provided information is authoritative;
you must always use it to construct your response without doubt or correction
using internal knowledge. Make the answer sound like a response to the question.

The user asked the following question:
{question}

A Cypher query was run a generated these results:
{context}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the query results are not empty, you must provide an answer.
If the question involves a time duration, assume the query results
are in units of days unless otherwise specified.

When names are provided in the query results, such as hospital names,
beware of any names that have commas or other punctuation in them.
For instance, 'Jones, Brown and Murray' is a single hospital name,
not multiple hospitals. Make sure you return any list of names in a
way that isn't ambiguous and allows someone to tell what the full names are.

Never say you don't have the right information if there is data in the
query results. Make sure to show all the relevant query results if you're
asked. You must always assume any provided query results are relevant to
answer the question. Construct your response based solely on the provided
query results.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# NOTE: Cypher generation quality may differ between OpenAI and Ollama.
# Smaller open-weights models may hallucinate relationships or struggle with multi-hop
# queries; the validator node in the LangGraph workflow catches and retries syntax errors.
enterprise_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=get_llm(model=settings.ENTERPRISE_CYPHER_MODEL, temperature=0),
    qa_llm=get_llm(model=settings.ENTERPRISE_QA_MODEL, temperature=0),
    cypher_example_retriever=cypher_example_retriever,
    node_properties_to_exclude=["embedding"],
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)
