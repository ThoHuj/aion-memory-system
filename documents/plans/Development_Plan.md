# **Engineering Subtasks: Bi-Temporal Knowledge Graph**

Based on the core objectives from Project Architecture Plan\_ Temporal Knowledge Graph(3).md and the roadmap in Development\_Plan(1).txt, here is the exhaustive task breakdown required to execute this project.

## **Phase 1: Minimum Viable Memory (Flat Retrieval)**

> **Detailed plans (source of truth):** Phase 1 is fully specified in the step plans below. This section is a summary index only.
>
> **Phase naming:** Architecture workflow phases (ingestion / retrieval / generation) differ from dev-plan delivery phases. See [Phase_1_Step_1_Plan.md §0](./Phase_1_Step_1_Plan.md#0-phase-naming-glossary).

| Step | Plan | Jira | Delivers |
|------|------|------|----------|
| 1 | [Phase_1_Step_1_Plan.md](./Phase_1_Step_1_Plan.md) | [VG-2](https://jira.atlassian.com/browse/VG-2) | Docker, FastAPI, Postgres + pgvector, `/health`, `/ready` |
| 2 | [Phase_1_Step_2_Plan.md](./Phase_1_Step_2_Plan.md) | [VG-3](https://jira.atlassian.com/browse/VG-3) | Flat ingest: LangGraph, embeddings, `POST /api/v1/ingest/flat` |
| 3 | [Phase_1_Step_3_Plan.md](./Phase_1_Step_3_Plan.md) | [VG-4](https://jira.atlassian.com/browse/VG-4) | Flat retrieve API, integration docs for external agent |

**This repo delivers:** ingest + retrieve REST APIs in Docker. **External agent repo** implements pre-fetch hook, prompt injection, and LLM orchestration.

**Phase 1 MVP loop (system-wide):** ingest via API → external agent calls retrieve on every turn → inject memory as user-message suffix. User chat is **not** auto-ingested in Phase 1; seed memories via the ingest API.

## **Phase 2: Bi-Temporal Graph & Extraction**

### **Step 1: Define strict Pydantic schemas enforcing entity/temporal constraints**

> * **2.1.1 Entity Schema:** Create a Pydantic model for entities (Entity), requiring attributes like entity\_id, name, type (Person, Organization, Concept, etc.), and aliases.  
> * **2.1.2 Temporal Constraints Schema:** Create a TemporalBounds schema containing valid\_from (datetime/ISO 8601 string) and valid\_to (datetime or null for ongoing).  
> * **2.1.3 Edge Schema:** Create a RelationEdge schema combining subject\_id, predicate (string), object\_id, and embedding the TemporalBounds.  
> * **2.1.4 Aggregated Output Schema:** Create an ExtractionResult schema that holds lists of Entities and RelationEdges.  
> * **2.1.5 Structured Output Prompting:** Configure the LLM calls to strictly output JSON matching the ExtractionResult schema (using OpenAI function calling or Gemini JSON mode).

### **Step 2: Execute database migrations for temporal edges**

> * **2.2.1 Migration Tooling Setup:** Initialize Alembic (or equivalent) in the project repository to handle structural DB changes safely.  
> * **2.2.2 Nodes Table Migration:** Write a migration script to create the nodes table (storing entities, their types, and optionally their embedded names/descriptions).  
> * **2.2.3 Sources Table Migration:** Create a sources table to track provenance (source\_id, raw\_text, file\_path).  
> * **2.2.4 Temporal Edges Migration:** Write a migration script for the edges table. Ensure columns include subject\_id (FK), object\_id (FK), predicate, valid\_from (TIMESTAMP), valid\_to (TIMESTAMP), and source\_id (FK).  
> * **2.2.5 Apply Migrations:** Execute the migrations against the PostgreSQL container to establish the relational graph schema.

### **Step 3: Upgrade LangGraph ingestion (timelines and pruning)**

> * **2.3.1 Entity Resolution Logic:** Build a function that checks if an extracted entity already exists in the nodes database (by exact name or semantic similarity) before inserting a new one, returning the existing ID to prevent graph fragmentation.  
> * **2.3.2 Graph Insertion Transaction:** Write an atomic database transaction (all-or-nothing) that inserts new nodes, inserts the source text into sources, and links the entities via the edges table with their valid\_from/valid\_to timestamps.  
> * **2.3.3 Graph Ingestion Endpoint:** Replace or upgrade the FastAPI /ingest route to use this new structured pipeline.  
> * **2.3.4 Temporal Pruning Query:** Update the retrieval SQL query. When selecting edges, add a strict WHERE clause: WHERE (valid\_to IS NULL OR valid\_to \>= CURRENT\_TIMESTAMP). (Or pass a specific timestamp to query "memory as it was on X date").

## **Phase 3: Hybrid Discovery & Algorithmic Pruning**

### **Step 1: Enable pg\_trgm and exact-match lexical indexing**

> * **3.1.1 Extension Migration:** Create an Alembic migration executing CREATE EXTENSION IF NOT EXISTS pg\_trgm;.  
> * **3.1.2 Lexical Index Creation:** Add a GIN index on the name column of the nodes table to vastly speed up string similarity searches (e.g., CREATE INDEX trgm\_idx ON nodes USING GIN (name gin\_trgm\_ops);).  
> * **3.1.3 Trigram Query Logic:** Write a Python function that executes a text similarity search in Postgres using the % operator or \<-\> distance operator from pg\_trgm.

### **Step 2: Implement Reciprocal Rank Fusion (RRF)**

> * **3.2.1 Query Entity Extraction:** Create a fast LLM step at the start of the retrieval process that parses the user's query into target entities (e.g., "What is John's dog's name?" \-\>  
>   $$ "John", "dog" $$  
>   ).  
> * **3.2.2 Parallel Search Execution:** Using Python's asyncio.gather, simultaneously run the semantic vector search and the lexical trigram search for the extracted entities.  
> * **3.2.3 RRF Algorithm Implementation:** Write an RRF function: loop through results from both searches, assign a score 1 / (k \+ rank) to each node (where k is typically 60), and sum the scores for nodes appearing in both lists.  
> * **3.2.4 Entry Node Selection:** Sort the combined results by the RRF score and slice the top N nodes. These are your graph entry points.

### **Step 3: Maximal Marginal Relevance (MMR) traversal logic**

> * **3.3.1 Graph Expansion Algorithm:** Write a traversal function (e.g., BFS) that starts at the RRF entry nodes and queries the database for active edges connected to these nodes.  
> * **3.3.2 Context Subgraph Assembly:** Translate the retrieved edges into readable natural language propositions (e.g., "Node A \-![][image1]  
>   \-\> Node B").  
> * **3.3.3 MMR Vectorization:** Embed the newly discovered proposition texts to compare their semantic meaning against the original query and against *already selected* propositions.  
> * **3.3.4 MMR Scoring Formula:** Implement the MMR math: Score \= (Lambda \* Query\_Similarity) \- ((1 \- Lambda) \* Max\_Similarity\_To\_Selected\_Nodes).  
> * **3.3.5 Dynamic Halting Condition:** Implement a loop that adds propositions based on the highest MMR score. Break the loop early if the MMR score falls below a threshold (meaning no new information is being gained) or if a strict token/character limit is hit. Return this optimized context to the consumer agent.

## **Phase 4: Multi-Modal Ingestion & Provenance**

### **Step 1: Mount local OS file system volume for binary storage**

> * **4.1.1 Docker Volume Update:** Update docker-compose.yml to include a bind mount for the FastAPI service (e.g., \- ./local\_storage:/app/storage).  
> * **4.1.2 Directory Initialization:** Add startup logic in FastAPI (@app.on\_event("startup")) to ensure the /app/storage directory exists.  
> * **4.1.3 Secure File Upload Endpoint:** Create a POST /upload/file route using FastAPI's UploadFile.  
> * **4.1.4 File Saving & Pathing:** Generate a secure UUID for the uploaded file, save it locally to the mounted volume, and return the absolute/relative local path. Add path sanitization to prevent directory traversal attacks.

### **Step 2: Construct the Two-Stage Vision pipeline**

> * **4.2.1 VLM Integration:** Configure OpenRouter/Client to utilize a Vision-Language Model (e.g., GPT-4o, Claude 3.5 Sonnet, or Gemini 1.5 Pro).  
> * **4.2.2 Stage 1 Prompt Engineering:** Write a specific system prompt instructing the VLM to generate a dense, factual text summary of the image/PDF tailored for database searchability.  
> * **4.2.3 File-to-Base64 Conversion:** Write a utility to read the local binary file and encode it to Base64 to pass it into the VLM payload.  
> * **4.2.4 Graph Integration Pipeline:** Feed the generated text summary directly into the Phase 2 extraction pipeline (Pydantic entity extraction).  
> * **4.2.5 Source Linking:** When saving the extracted edges to the database, populate the sources table with file\_path \= \[UUID\_path\] and file\_type \= \[mime\_type\], linking the generated source\_id to the new edges.

### **Step 3: Dedicated provenance route**

> * **4.3.1 API Provenance Endpoint:** Create a GET /provenance/{source\_id} endpoint in FastAPI.  
> * **4.3.2 Database Lookup:** Write a query to fetch the row from the sources table matching the ID.  
> * **4.3.3 Response Handling (Text vs Binary):** If the source is text, return the raw snippet as a JSON response. If the source is an image/PDF, use FastAPI's FileResponse to stream the actual file back to the consumer agent.  
> * **4.3.4 Consumer Agent Tooling:** Provide the consumer agent with a tool/function call (e.g., fetch\_source(source\_id)) that it can autonomously trigger if the user asks "Show me the original document" or "How do you know that?".

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAqCAYAAAAOCwd9AAABt0lEQVR4Xu3abU3FQBAF0GrAAhqwgAUsYAELOEACEnCAAxxgAAGQm3TCMLw+Ej5C0ndO0nR3Z9/Cz5vZLgsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/qtY2vx/y7csblOj5b5991OxcAAE5JwtDDWPtJuCq/cUYk7N3PRQCAU3IoWPW1m/U9u1zn61oCVVf7ng6sdVfL5/U5T4cu5+S99XcAAHavri7zJAT1sJbOVuZ5Llqt73lc3wlUNU69Atbs4CXo1fxleb82zTiel4/nz0CZfXX2rAEA7NJXoSf1dMNiBqWErQpax0LWVq26d3VGjSv4Rd/fv69zVQoAnIQErhmupllPmKqg1W2FsjmftThWT8etpNbDHQDA7iUA1ZXklv4tWiSs9d/kqjRm6KpQly5YrkGrPgPZDI01zm/u1neZ3TcdNgBg1xKM6tmy9WF/glM6Xz0w5boyaxXOMo8Eut4ly/Vqft/DYu2p8Nb/p+ytUBgJbHn6mQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPA33gDIcWLG3F9IugAAAABJRU5ErkJggg==>