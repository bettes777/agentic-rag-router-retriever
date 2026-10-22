# Agentic RAG: Router-Retriever System with PDF and Web Search Tools

A two-agent CrewAI system that answers questions from either a static PDF knowledge base or the live web, choosing the source per question and logging every step.

Built as a course-end project for the **Post Graduate Program in Applied Agentic AI** (Virginia Tech and Simplilearn).

## Files

| File | Purpose |
|---|---|
| `agentic_rag_router_retriever.ipynb` | Working notebook: setup, tools, agents, demo runs, trace logs |
| `README.md` | This document |
| `requirements.txt` | Python packages |
| `.env.example` | Names of the credentials the notebook needs (no real values) |

Running the notebook also creates `agent_trace_log.jsonl`, the interaction log (one JSON event per line). The trace from the original run is shown in Step 12 of the notebook.

**Knowledge base:** the 4-page Executive summary of the *Preliminary Report of the Independent International Scientific Panel on AI* (United Nations, July 2026), file `un_scientific_panel_ai_preliminary_report_summary.pdf`. The report is an evidence-based assessment of AI capabilities, risks, impacts, governance and evidence gaps. The summary split into 9 text chunks for indexing.

**To run:**
1. Install the packages: `pip install -r requirements.txt` (Python 3.12 was used).
2. Copy `.env.example` to `.env` and add your own Azure OpenAI and Tavily credentials.
3. Download the PDF (see **Source document** below) and save it in the project folder as `un_scientific_panel_ai_preliminary_report_summary.pdf`. Any file name starting with `un_scientific_panel_ai_preliminary_report` anywhere under the notebook's folder will be found.
4. Run all cells in order.

**Source document:** the PDF is not stored in this repository. The Executive summary (4 pages, July 2026) of the *Preliminary Report of the Independent International Scientific Panel on AI* is available from the [UN Scientific Panel report page](https://www.un.org/independent-international-scientific-panel-ai/en/preliminary-report), along with the full report.

## 1. System architecture

```
                 User question
                       │
                       ▼
        ┌──────────────────────────────┐
        │  ROUTER AGENT ("Query Router")│   no tools
        │  Output: RouteDecision        │   {route, reason, search_query}
        └──────────────┬───────────────┘
                       │  context=[router_task]
                       ▼
        ┌──────────────────────────────┐
        │  RETRIEVER AGENT             │
        │  follows the route:          │
        │   PDF ─► pdf_search          │──► PDFSearchTool (UN report)
        │   WEB ─► web_search          │      └─ fallback: FAISS index
        │   LLM ─► no tool             │──► TavilySearchResults
        └──────────────┬───────────────┘
                       ▼
        Grounded answer + "Sources:" list

  step_callback / task_callback ──► agent_trace_log.jsonl
```

| Component | Technology |
|---|---|
| Orchestration | CrewAI, `Process.sequential` |
| LLM | Azure OpenAI `gpt-5-mini` via CrewAI `LLM` (LiteLLM) |
| PDF retrieval | `crewai_tools.PDFSearchTool` over the UN report (Azure embeddings), FAISS fallback |
| Web retrieval | `TavilySearchResults` (LangChain) wrapped as a CrewAI `BaseTool` |
| Structured routing output | Pydantic `RouteDecision` via `output_pydantic` |
| Logging | CrewAI `step_callback` and `task_callback` → JSONL file + pandas tables |

## 2. Agents: logic and responsibilities

**Router Agent (Query Router)**
- Input: the user's natural-language question.
- Logic: applies routing rules written into its backstory:
  - **PDF**: questions about what the UN Scientific Panel's report says, finds or recommends (capabilities, risks, economic/security/environmental/rights impacts, governance, evidence gaps, next steps).
  - **WEB**: questions needing current or recent information (anything after the report's July 2026 publication), or topics outside its scope.
  - **LLM**: general-knowledge or definitional questions that need no retrieval.
- Output: a `RouteDecision` object (`route`, `reason`, `search_query`). It has no tools and never answers the question itself, which keeps its job narrow and its decision auditable.

**Retriever Agent (Retriever and Answer Writer)**
- Input: the question plus the router's decision (passed as task context).
- Logic: calls only the tool named by the router (or no tool for LLM), then writes a 150–250 word answer grounded in the retrieved text.
- Output: an answer starting with `Route used: ...` and ending with a `Sources:` list of report pages or URLs.

## 3. Coordination flow

1. `ask(question)` assigns a run ID and logs `question_received`.
2. A fresh crew is built and started with `inputs={"question": ...}`.
3. **Router task** runs; its structured output is logged as `route_decision`.
4. **Retriever task** receives the router output through `context=[router_task]`, calls the selected tool (each call logged with the backend that answered it), and produces the final answer.
5. The final answer is logged and printed; Step 12 of the notebook shows a routing summary table, the full event trace, and a narrative per run.

## 4. Demo results

All four questions in the final run were routed as intended:

| Run | Question | Router decision | Tool calls by Retriever |
|---|---|---|---|
| 1 | Main evidence gaps in the UN Panel's report | **PDF** | `pdf_search` × 5 (FAISS) |
| 2 | This week's EU AI regulation news | **WEB** | `web_search` × 1 (TavilySearchResults) |
| 3 | What is retrieval-augmented generation? | **LLM** | none |
| 4 | What the report says about agentic AI as a governance step change | **PDF** | `pdf_search` × 4 (FAISS) |

Each answer began with `Route used: ...`, matching the router's decision, and in every run the Retriever used only the tool the router chose (or none, for LLM). The router's one-sentence reason for each decision is in the Step 12 tables, for example: Run 2 was sent to WEB because "this week's news" is time-sensitive and falls after the July 2026 report.

Run 4 appears twice under `question_received` in the trace because that cell was run twice; only the completed run produced an answer.

## 5. Why CrewAI

- **Role-based agents** (role, goal, backstory, tools) map directly to the Router/Retriever design the project describes.
- **Built-in task chaining** (`context=[...]`) gives a clear, explicit input-output handoff between agents without custom plumbing.
- **Structured outputs** (`output_pydantic`) make the routing decision machine-readable and easy to log.
- **Callbacks and verbose mode** support the traceability requirement.
- **Native tool ecosystem** (`crewai_tools.PDFSearchTool`) and easy wrapping of LangChain tools such as Tavily.
- It was also the framework used in the course modules, so the setup (Azure LLM, prompt-shield workaround) was already proven.

Alternatives considered: LangGraph offers finer control over branching but needs more code for the same result; a single LangChain agent would blur the router and retriever roles the project asks to separate.

## 6. Challenges and trade-offs

**Sequential routing vs. programmatic routing.** The router's decision is passed to a retriever that holds both tools; the retriever is *instructed* to use only the chosen one. This is simpler and keeps the whole flow inside one CrewAI crew, but compliance depends on the LLM following instructions. A stricter design would parse the decision in Python and give the retriever only the selected tool. The trace log makes any deviation visible: every tool call logs which backend answered it. In the final run the Retriever followed the route every time.

**PDFSearchTool with Azure embeddings.** CrewAI's documentation lists Azure as an embedding provider but gives no complete Azure example, and community reports describe it failing. The notebook therefore tries `PDFSearchTool` first and falls back automatically to a FAISS index built with `AzureOpenAIEmbeddings`. The backend actually used is logged, so the result is transparent either way.

*What happened in the final run:* `PDFSearchTool` reached the Azure API, but indexing failed with `RateLimitError: 429 – Token limit will exceed based on estimated request tokens` during the upsert step, and the system switched to the FAISS fallback. The exact cause is not confirmed. One factor: in that run `PDFSearchTool` was still configured with an incorrect embedding deployment name (see the next point); the gateway appears to have applied its rate limit before checking the name. So this run does not show whether `PDFSearchTool` would work once correctly configured. All PDF answers in the demo come from the FAISS backend, as the trace log's `backend` entries show. With the corrected name and an unthrottled moment, the same code would try `PDFSearchTool` first again with no changes.

**Finding the embedding deployment name.** The course gateway's embedding deployment name was not documented. The first guess (`text-embedding-3-small`) returned `404 DeploymentNotFound`. A small diagnostic cell (after Step 3) tested common names and found `text-embedding-ada-002`. Because the notebook kernel kept the old value in memory, an extra cell before Step 8 sets the correct name explicitly and clears the cached index before the first PDF search. Both cells were kept in the notebook as evidence of the troubleshooting.

**Embedding rate limits.** The Azure gateway throttled embedding requests in the earlier RAG project. The FAISS fallback embeds in batches of 5 with a 15-second pause and saves the index to disk, so the build happens only once. For the 9-chunk summary PDF it took well under a minute; the earlier three-report RAG project (309 chunks) took 10–15 minutes.

**Azure prompt shield.** CrewAI's default "force final answer" prompt wording triggered Azure's jailbreak filter. The notebook overrides that one internal prompt string. This relies on a CrewAI internal (`I18N_DEFAULT._prompts`) and may break on a future CrewAI upgrade.

**Empty LLM responses.** In the course demo, `gpt-5-mini` once returned an empty response with `max_completion_tokens=1500`. The limit was raised to 4000. A possible explanation is that reasoning models use part of the token budget internally, but this was not confirmed.

**Tavily tool naming.** The project specifies `TavilySearchResults`. LangChain has been moving Tavily into the separate `langchain-tavily` package (`TavilySearch`), so the notebook uses `TavilySearchResults` when available and falls back to `TavilySearch` otherwise, reporting which one ran.

**Overlapping scope between routes.** The report itself discusses AI governance, so a question such as "what is happening with UN AI governance?" could reasonably go to PDF or WEB. The router's rules resolve this by time: what the Panel *says or found* goes to PDF; anything *current* or after July 2026 goes to WEB. Ambiguous questions remain the main source of routing errors, which is why the router's reason is logged.

**Grounding vs. breadth.** Answers are limited to 150–250 words and required to cite sources. This favors accuracy and traceability over exhaustive answers.

**Repeated tool calls and latency.** The Retriever was told to refine its search query at most once, but on PDF questions it called `pdf_search` 4–5 times before answering. The extra searches gathered more evidence but did not strictly follow the instruction, and they were the main reason a PDF question took a few minutes. In Run 1 the Retriever reached its `max_iter=5` limit, and CrewAI then requested a final answer. Together with at least two LLM calls per question (router and retriever) and the gateway's rate limits, each demo question took roughly 1–4 minutes. Routing adds latency compared with a single RAG chain, in exchange for choosing the right source per question. Lowering `max_iter` further (for example to 2 or 3) would cap the number of searches more tightly, at the cost of less evidence per answer.

**Trace capture limits.** The logger uses two CrewAI hooks. `task_callback` and the custom tool wrappers recorded reliably: task completions, routing decisions, final answers, and each tool call with its backend. `step_callback` captured agents' final answers but not the intermediate tool-call steps, likely because this CrewAI version passes step objects in a different shape than the logger expected. Tool usage is therefore evidenced by the backend entries logged inside each tool rather than by `step_callback`. The full step-by-step detail is still visible in the notebook's verbose output.

**Grounding is not guaranteed.** Checking the answers against the source PDF found two issues worth noting:
- In Run 4, the answer says the length of certain software tasks AI systems can complete is "halving every 4–7 months". The report says it is *doubling*. The routing and retrieval were correct, but the model misstated a retrieved fact, and the citation format alone did not catch it.
- In Run 3 (LLM route, no retrieval), the answer still ended with a `Sources:` list of academic papers, cited from the model's own knowledge rather than from anything retrieved. The answer itself noted that it had not retrieved them.

This is a reason to add an answer-checking step (see improvements below).

**Summary instead of full report.** Only the 4-page Executive summary was indexed, to keep embedding within the gateway's rate limits. In Run 4, the answer correctly reported that the retrieved text does not use the phrase "governance step change". The full report does use it as a section heading, so the summary-only index limits what the PDF route can find.

## 7. Possible improvements

- Enforce the route in code (programmatic routing) or with CrewAI Flows.
- Capture intermediate tool steps in `step_callback` by adapting it to the step-object format of the installed CrewAI version.
- Add a "both" route that combines PDF and web evidence for questions that need each.
- Add an evaluator agent that checks the answer against the retrieved passages before returning it (this would have caught the "halving"/"doubling" error in Run 4).
- Index the full report once rate limits allow, instead of only the Executive summary.
- Cache web results and reuse the vector index across sessions.
