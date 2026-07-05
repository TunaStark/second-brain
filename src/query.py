"""Query pipeline: interactive RAG chat over your second brain.

Run:  python -m src.query               (interactive)
      python -m src.query "question"    (one-shot)
"""

import logging
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rich.console import Console
from rich.panel import Panel

from src import config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
console = Console()

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


def build_chain():
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
        console.print("[red]Vector store is empty. Run `python -m src.ingest` first.[/red]")
        sys.exit(1)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVAL_K,
            "fetch_k": config.RETRIEVAL_FETCH_K,
            "lambda_mult": config.RETRIEVAL_LAMBDA,
        },
    )
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        num_ctx=config.LLM_NUM_CTX,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer(chain, retriever, question: str) -> None:
    sources = retriever.invoke(question)

    console.print()
    try:
        for token in chain.stream(question):
            console.print(token, end="")
    except Exception as exc:
        console.print(f"\n[red]LLM call failed: {exc}[/red]")
        console.print(
            f"[yellow]Check Ollama is running and `ollama pull {config.LLM_MODEL}` was done.[/yellow]"
        )
        return
    console.print("\n")

    seen, lines = set(), []
    for doc in sources:
        src = doc.metadata.get("source", "unknown")
        kind = doc.metadata.get("source_type", "note")
        if src not in seen:
            seen.add(src)
            lines.append(f"• [{kind}] {src}")
    if lines:
        console.print(Panel("\n".join(lines), title="Sources", border_style="dim"))


def main() -> None:
    chain, retriever = build_chain()

    # One-shot mode
    if len(sys.argv) > 1:
        answer(chain, retriever, " ".join(sys.argv[1:]))
        return

    # Interactive mode
    console.print(
        Panel(
            f"[bold]Second Brain[/bold] — local RAG over your notes & bookmarks\n"
            f"LLM: {config.LLM_MODEL} | Embeddings: {config.EMBEDDING_MODEL}\n"
            f"Type your question. 'exit' to quit.",
            border_style="cyan",
        )
    )
    while True:
        try:
            question = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        answer(chain, retriever, question)
    console.print("[dim]bye.[/dim]")


if __name__ == "__main__":
    main()
