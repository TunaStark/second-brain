"""Chainlit web chat: RAG over your second brain.

Run:  chainlit run src/query.py -w
"""

import json
import re
import sys
from pathlib import Path

from flashrank import Ranker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chainlit as cl
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src import config

# chainlit persists inline elements under .files/<session>/ but does not create the parent
(config.PROJECT_ROOT / ".files").mkdir(exist_ok=True)

MAX_HISTORY_MESSAGES = 8  # keep prompt inside LLM_NUM_CTX
MAX_SOURCE_NAME_LEN = 50  # element names longer than this overflow the source cards

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


def _bm25_tokenize(text: str) -> list[str]:
    # lowercase + split on non-word chars, so "web3" hits inside URLs and titles
    return re.findall(r"\w+", text.lower())


def _source_name(doc: Document) -> str:
    """Short, card-safe label: prefer the clean title over the raw URL/path."""
    kind = doc.metadata.get("source_type", "note")
    label = doc.metadata.get("title") or doc.metadata.get("source", "unknown")
    name = f"[{kind}] {label}"
    if len(name) > MAX_SOURCE_NAME_LEN:
        name = name[: MAX_SOURCE_NAME_LEN - 3] + "..."
    return name


def _source_content(doc: Document) -> str:
    """Card body: full clickable URL/path plus the retrieved snippet."""
    src = doc.metadata.get("source", "unknown")
    link = f"[{src}]({src})" if src.startswith(("http://", "https://")) else src
    return f"**Source:** {link}\n\n{doc.page_content}"


def _load_docstore() -> list[Document]:
    """Chunks persisted by ingest — the BM25 corpus."""
    if not config.DOCSTORE_PATH.is_file():
        raise RuntimeError("Docstore missing. Run `python -m src.ingest` first.")
    docs = []
    with open(config.DOCSTORE_PATH, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["page_content"], metadata=rec["metadata"]))
    return docs


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

    # Dense: semantic depth (bge-m3). Sparse: exact keyword hits (BM25).
    dense = vectorstore.as_retriever(search_kwargs={"k": config.HYBRID_DENSE_K})
    sparse = BM25Retriever.from_documents(
        _load_docstore(),
        k=config.HYBRID_SPARSE_K,
        preprocess_func=_bm25_tokenize,
    )
    hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=config.ENSEMBLE_WEIGHTS)

    # Rerank the ~20 merged candidates down to the best 5 for the LLM.
    reranker = FlashrankRerank(
        client=Ranker(model_name=config.RERANK_MODEL),
        top_n=config.RERANK_TOP_N,
    )
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)


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
            f"LLM: `{config.LLM_MODEL}` | Embeddings: `{config.EMBEDDING_MODEL}` | "
            f"Hybrid (dense+BM25) → FlashRank top-{config.RERANK_TOP_N}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")
    if history is None:
        history = []

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
        if src in seen:
            continue
        seen.add(src)
        name = _source_name(doc)
        if name in names:  # element names must be unique for inline references
            name = f"{name[: MAX_SOURCE_NAME_LEN - 4]}({len(names)})"
        names.append(name)
        elements.append(cl.Text(name=name, content=_source_content(doc), display="inline"))
    if elements:
        await msg.stream_token("\n\n**Sources:** " + " · ".join(f"`{n}`" for n in names))
        msg.elements = elements
    await msg.update()

    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=msg.content))
    cl.user_session.set("history", history[-MAX_HISTORY_MESSAGES:])
