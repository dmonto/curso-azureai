import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv("../.env")

endpoint = os.getenv("AOAI_ENDPOINT_SECONDARY")
api_key = os.getenv("AOAI_API_KEY_SECONDARY")
api_version = os.getenv("AOAI_API_VERSION", "preview")
deployment = os.getenv("AOAI_DEPLOYMENT_SECONDARY", "gpt-4o")

model = ChatOpenAI(
    model=deployment,
    base_url=f"{endpoint}/openai/v1/",
    api_key=api_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente técnico conciso."),
    ("human", "Explica en dos frases qué aporta LangChain como marco de composición."),
])

chain = prompt | model | StrOutputParser()

result = chain.invoke({})
print(result)