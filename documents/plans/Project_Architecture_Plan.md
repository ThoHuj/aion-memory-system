# **Project Architecture Plan: Bi-Temporal Knowledge Graph for Agentic Memory**

**Target:** AI Engineering "Pass with Distinction" Assignment

**Date:** September 2026

**Core Objective:** Design and implement a highly token-efficient, context-aware, and time-accurate long-term memory system for an LLM agent, maximizing prompt cache retention and supporting verifiable provenance.

## **1\. Executive Summary**

Modern LLM agents struggle with context window bloat, prompt caching invalidation, and the "flatness" of vector-only memory when facts change over time. This project solves these issues by implementing a **Bi-Temporal Knowledge Graph** supported by a "static-first, dynamic-last" prompt injection strategy. The agent will autonomously extract facts, track their validity over time, ingest multimodal files, and retrieve context using a hybrid Semantic \+ Lexical graph search, all while allowing the user to request explicit source provenance on-demand.

## **2\. Technology Stack**

> * **Microservice Architecture:** Docker \+ FastAPI \- The memory engine is deployed as a standalone, decoupled microservice rather than baked directly into the consumer agent.  
> * **Orchestration & Control Flow:** LangGraph (Python) \- Operates inside the microservice boundary to manage background extraction, cyclical reasoning, and deterministic temporal traversals.  
> * **Database:** PostgreSQL (with pgvector and pg\_trgm extensions) \- Serves as the unified storage layer for relational data, graph edges, vector embeddings, and full-text trigram indexing.  
> * **Object Storage:** Local File System (OS File Manager) \- Stores binary files (PDFs, raw images) locally to ensure fast I/O, data privacy, and a self-contained containerized deployment without cloud dependencies.  
> * **Data Validation:** Pydantic \- Enforces strict JSON schemas for entity and temporal extraction.  
> * **Generative Models:**  
  * Fast LLM (e.g., Gemini Flash) for strict background entity extraction.  
  * Vision-Language Model (VLM) for the Two-Stage Vision pipeline.  
  * Core reasoning LLM (Consumer Agent) for user interaction.

## **3\. Deployment & Integration Strategy**

> * **Decoupled Engine:** The memory system is an isolated API, owning its own database migrations, file storage, and extraction logic. The **consumer agent lives in a separate repository** and integrates exclusively via REST (or MCP wrappers later) — it is not part of the memory-service codebase.  
> * **Deterministic Pre-Fetch Pattern:** Instead of relying on the consumer agent to autonomously decide when to use a tool to search its memory (which increases latency and hallucination risk), the consumer agent's orchestrator (in the external agent repo) contains a mandatory pre-generation hook. It intercepts the user's query, fetches context deterministically from the FastAPI service, and injects it before the core LLM generates a response.  
> * **MCP Future-Proofing:** While initially deployed as a REST microservice for deterministic pre-fetching, the endpoints are designed so they can be easily wrapped into standard Model Context Protocol (MCP) tools (e.g., retrieve\_temporal\_context, fetch\_source\_quote) later on.

## **4\. Core Workflows (How the System Works)**

### **Phase 1: Ingestion & Extraction (Background Pipeline)**

When a user interacts with the agent or uploads a file, the raw input is processed asynchronously:

> 1. **Text (Pydantic Extraction):** A fast LLM pipeline uses Pydantic to extract structured, strict schemas containing specific target entities, relationships, and the timestamp of the event.  
> 2. **Images/PDFs:**  
   * The raw file is saved securely to a designated directory on the local file system.  
   * A VLM generates a lightweight text summary (Ingestion Stage 1\) to make the file searchable via embeddings.  
> 3. **Storage:** The extracted fact is saved in PostgreSQL as a temporal edge: (Subject)-\[Relation, valid\_from, valid\_to, source\_id\]-\>(Object).  
> 4. **Provenance:** The raw text snippet or absolute local file path is saved in a separate sources table, linked to the edge via source\_id.

### **Phase 2: Retrieval (Hybrid Search & Graph Traversal)**

When the pre-fetch hook intercepts the user's query:

> 1. **Structured Entity Resolution:** The raw query is passed through a fast LLM to extract the targeted entities (e.g., transforming "What project was I on last summer?" to Entities: \["Project", "Summer 2026"\]), preventing asymmetric retrieval mismatch.  
> 2. **Hybrid Entry Node Discovery:** These strict entities are simultaneously fed into a semantic scan (pgvector) and a lexical exact-match scan (pg\_trgm). The scores are combined via Reciprocal Rank Fusion (RRF) to discover highly accurate graph entry points.  
> 3. **Temporal Traversal:** The graph expands outward from the entry nodes. It explicitly checks the valid\_to column of every edge. Any edge with a timestamp in the past is pruned.  
> 4. **Reward-Based Halting (MMR):** The traversal algorithm uses Maximal Marginal Relevance to calculate Information Gain. Conceptually redundant branches are terminated early, drastically reducing token bloat.

### **Phase 3: Generation & Lazy-Loading Context**

> 1. **Cache-Preserving Injection:** The retrieved active subgraph is returned to the consumer agent and injected at the very bottom of the LLM prompt (the suffix). The system prompt and tool definitions remain static at the top, perfectly preserving the LLM's KV cache.  
> 2. **On-Demand Provenance:** The agent receives the facts and their lightweight source\_ids. If the user asks for proof, the agent dynamically runs a query to pull the raw diary entry or triggers a VLM to re-read a specific image directly from the local file system based on the user's exact context (Two-Stage Vision).

## **5\. Discarded Architectural Choices (and Why)**

> * **Universal Hybrid Memory Middleware (e.g., Mem0):**  
  * *Discarded Because:* Often flattens temporal history into synthesized text summaries, losing the precise deterministic control over timelines that a custom bi-temporal graph provides.  
> * **Agent OS (Letta / MemGPT Pattern):**  
  * *Discarded Because:* Relying on the LLM to autonomously manage "Core" vs "Archival" memory via tool calls increases latency per turn and risks the LLM accidentally evicting crucial context. The Deterministic Pre-Fetch pattern is safer and faster.  
> * **Native Graph Database (Neo4j):**  
  * *Discarded Because:* PostgreSQL natively supports vectors (pgvector), trigrams (pg\_trgm), relational data (the sources table), and graph traversal in a unified service.  
> * **Database File Storage (bytea):**  
  * *Discarded Because:* Storing heavy PDFs and images directly in PostgreSQL bloats the database and slows down graph traversal. Utilizing the local file system for binary storage with reference paths in Postgres is a strictly superior engineering pattern that maintains high database performance while keeping deployment localized.

## **6\. "Pass with Distinction" Features**

This project earns top marks by implementing:

> 1. **Hybrid Retrieval Precision:** Solving asymmetric retrieval by combining Pydantic entity extraction with fused semantic (pgvector) and lexical (pg\_trgm) database searches.  
> 2. **KV Cache Optimization:** Structuring memory retrieval to explicitly respect prefix caching rules (static-first, dynamic-last).  
> 3. **Bi-Temporal State Management:** Deterministically tracking the evolution of facts over time without relying on LLM guesswork.  
> 4. **Algorithmic Token Optimization:** Implementing MMR vector math to dynamically halt graph search.  
> 5. **Decoupled Architecture:** Deploying the memory engine as an independent microservice that supports a deterministic pre-fetch pattern, cleanly separating the stateful graph from the stateless agent.