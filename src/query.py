"""Chainlit web chat: RAG over your second brain.

Run:  chainlit run src/query.py -w
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src import config

MAX_HISTORY_MESSAGES = 8  # keep prompt inside LLM_NUM_CTX

SYSTEM_PROMPT = """\
You are "Second Brain", the user's personal knowledge assistant.
Answer ONLY from the provided context (the user's own notes and bookmarks).
Rules:
- If the context does not contain the answer, say so plainly. Never invent facts.
- Answer in the same language as the question.
- Be concise and well-structured. Use bullet points where helpful.

Context:
{context}
"""


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_retriever():
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.STORAGE_DIR),
    )
    if not vectorstore._collection.count():
        raise RuntimeError("Vector store is empty. Run `python -m src.ingest` first.")

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVAL_K,
            "fetch_k": config.RETRIEVAL_FETCH_K,
            "lambda_mult": config.RETRIEVAL_LAMBDA,
        },
    )


def build_chain():
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        num_ctx=config.LLM_NUM_CTX,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


@cl.on_chat_start
async def on_chat_start():
    try:
        retriever = build_retriever()
    except RuntimeError as exc:
        await cl.Message(content=f"⚠️ {exc}").send()
        return

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("chain", build_chain())
    cl.user_session.set("history", [])

    await cl.Message(
        content=(
            f"**Second Brain** — local RAG over your notes & bookmarks\n\n"
            f"LLM: `{config.LLM_MODEL}` | Embeddings: `{config.EMBEDDING_MODEL}`"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")

    if retriever is None or chain is None:
        await cl.Message(content="Not ready — vector store missing. Ingest, then start a new chat.").send()
        return

    async with cl.Step(name="retrieval", type="retrieval") as step:
        docs = await retriever.ainvoke(message.content)
        step.output = "\n".join(
            f"[{i}] {d.metadata.get('source', 'unknown')}" for i, d in enumerate(docs, 1)
        )

    msg = cl.Message(content="")
    try:
        async for token in chain.astream(
            {
                "context": format_docs(docs),
                "question": message.content,
                "history": history,
            }
        ):
            await msg.stream_token(token)
    except Exception as exc:
        await cl.Message(
            content=(
                f"LLM call failed: `{exc}`\n\n"
                f"Check Ollama is running and `ollama pull {config.LLM_MODEL}` was done."
            )
        ).send()
        return

    elements, names, seen = [], [], set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        kind = doc.metadata.get("source_type", "note")
        if src in seen:
            continue
        seen.add(src)
        name = f"[{kind}] {src}"
        names.append(name)
        elements.append(cl.Text(name=name, content=doc.page_content, display="inline"))
    if elements:
        await msg.stream_token("\n\n**Sources:** " + " · ".join(f"`{n}`" for n in names))
        msg.elements = elements
    await msg.update()

    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=msg.content))
    cl.user_session.set("history", history[-MAX_HISTORY_MESSAGES:])
