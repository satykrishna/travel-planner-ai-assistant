# Travel Planner AI Assistant

A notebook-first, multi-agent travel planner built with LangGraph, LangChain, OpenAI, Tavily, and SerpAPI. It classifies travel requests, dispatches flight/hotel and destination-research specialists, executes structured tool calls in normal Python graph nodes, and synthesizes their findings.

The source is [Multi_Agent_Travel_Planner.ipynb](Multi_Agent_Travel_Planner.ipynb). The repository also includes a FastAPI web app with a browser interface and Vercel configuration.

## What it does

- Searches Google Flights through SerpAPI, applies maximum-stop constraints, and reports the number of returned flight options.
- Searches Google Hotels or vacation rentals, validates stay dates and guest counts, and sorts returned properties by price by default.
- Delegates detailed itineraries to a deep research agent with web search.
- Uses `Send` to dispatch one or both specialist subgraphs in parallel.
- Keeps separate HumanMessage, AIMessage, and ToolMessage histories for each specialist, including tool-call IDs.
- Checks destination-local flight arrival dates against quoted hotel dates before synthesis.
- Prints a readable final answer and a trace of routing decisions and tool calls.

## Workflow

```mermaid
flowchart TD
    START --> Classifier
    Classifier -->|Send: flights / hotels| Search
    Classifier -->|Send: research / itinerary| Scout
    subgraph Search[Search specialist]
        SearchModel[search_agent_node] -->|tool calls| SearchTools[search_tool_node]
        SearchTools --> SearchModel
    end
    subgraph Scout[Travel scout specialist]
        ScoutModel[travel_scout_agent_node] -->|tool calls| ScoutTools[travel_scout_tool_node]
        ScoutTools --> ScoutModel
    end
    Search -->|completed findings| Synthesis
    Scout -->|completed findings| Synthesis
    Synthesis --> END
```

The classifier chooses `search_agent`, `travel_scout_agent`, or both. The specialist subgraphs finish their complete model/tool loops before the parent merges results and runs synthesis once. A request selecting only one specialist does not wait for an unselected branch.

Tools are advertised with `llm.bind_tools(...)`; this only supplies their schemas. The executor nodes resolve tool-name strings through allowlisted dictionaries, call `.invoke(tool_call['args'])`, and create matching `ToolMessage` objects. These executor nodes are ordinary graph functions, not `ToolNode` instances.

## Requirements

- Python 3.11 or 3.12; the original notebook was developed in PyCharm with Python 3.12.
- An OpenAI API key, a Tavily API key, and a SerpAPI key with access to Google Flights / Google Hotels searches.
- JupyterLab or PyCharm with notebook support.

The main agent dependencies in `requirements.txt` are pinned to versions observed in the source notebook's working environment. Model calls and search calls may incur charges or consume provider quotas.

## Setup

```bash
git clone https://github.com/satykrishna/travel-planner-ai-assistant.git
cd travel-planner-ai-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate` instead. Add your own values to `.env`:

```dotenv
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
SERPAPI_API_KEY=your_serpapi_key
```

Run the kernel from this repository's root so the `.env` cell can locate the file. Existing process environment variables take precedence. Restart the kernel after changing previously loaded values. Never commit `.env`, credentials, or sensitive notebook output.

```bash
jupyter lab Multi_Agent_Travel_Planner.ipynb
```

In PyCharm, open the repository, select its Python interpreter, and open the notebook.

## Running the notebook

1. Install dependencies and load `.env`.
2. Run the definition cells in order: model and tools, prompts, Pydantic schemas, shared state, classifier, specialist nodes, synthesizer, and combined graph.
3. Keep the default flags disabled while exploring definitions:
   ```python
   RUN_LIVE_EXAMPLES = False
   RUN_INTERACTIVE_CHAT = False
   RUN_REFRESH_SYNTHESIS = False
   ```
4. Enable only the live examples you intend to execute. Interactive chat loops are separately guarded so they cannot block a normal run of the definition cells.

Sections 1–16 demonstrate the individual tools and earlier standalone agents. Sections 17–24 define the shared-state workflow. The combined graph is `travel_planner_workflow`; the old search-only graph is not required.

To run the final workflow directly after its definitions:

```python
query = (
    "Plan a trip for two adults from YYZ to CDG, departing December 10, 2026 "
    "and returning December 13, 2026. Find flights with at most one stop, "
    "one hotel room for December 11–13, and a relaxed sightseeing itinerary. "
    "Respect actual arrival dates and do not book anything."
)

travel_planner_result = travel_planner_workflow.invoke(
    {
        "user_query": query,
        "messages": [],
        "agent_results": [],
        "search_agent_messages": [],
        "travel_scout_agent_msgs": [],
    },
    config={"recursion_limit": 80},
)

print(travel_planner_result["synthesized_answer"])
display_travel_planner_result(travel_planner_result)
```

Update the example dates when they are no longer in the future. Start every new workflow run with fresh specialist histories and results. This example performs live requests; it is not an offline test.

To inspect nested progress instead of waiting for `.invoke()`:

```python
for namespace, update in travel_planner_workflow.stream(
    {
        "user_query": query,
        "messages": [],
        "agent_results": [],
        "search_agent_messages": [],
        "travel_scout_agent_msgs": [],
    },
    config={"recursion_limit": 80},
    stream_mode="updates",
    subgraphs=True,
):
    print(namespace, update)
```

Trace output may contain travel requests and provider results. Review it before sharing or committing.

## Date consistency and search limitations

- Overnight flights can arrive the day after departure. The date checker compares each returned option's destination-local arrival with the hotel dates actually searched.
- Explicit hotel dates are preserved. A hotel stay starting before arrival is flagged as a possible extra night, rather than silently changed.
- Different hotel dates require a new search; quotes must not be relabeled or prorated for a shorter stay.
- The synthesizer is instructed to keep activities within the arrival/return window. Its prose is still model-generated and should be reviewed; the deterministic checks do not validate every itinerary sentence.
- The hotel API does not expose a documented room-count filter. This notebook records `no_of_rooms` and rejects multi-room requests rather than claiming verified multi-room availability.
- `airbnb_type=True` means Google Vacation Rentals, not Airbnb-only listings. Children require one age entry each.
- Hotel counts and sorting cover the returned page. Flight counts represent returned itinerary options, not every available flight.
- Search-result flight legs do not necessarily include a selected return itinerary. The workflow does not book or confirm reservations.
- Unknown tool names are skipped by the simple manual executors, and invocation errors propagate. This is a learning notebook, not a hardened public API.
- `KeyboardInterrupt` means execution was stopped. Deep research may take longer than direct search calls; use progress output or streaming to inspect it.

## Repository files

- `Multi_Agent_Travel_Planner.ipynb`: clean notebook copy with no saved outputs or execution counts.
- `requirements.txt`: notebook dependencies.
- `.env.example`: empty credential placeholders.
- `.gitignore`: excludes local secrets, environments, checkpoints, and IDE files.
- `validate_notebook.py`: checks notebook structure, Python-cell syntax, required workflow definitions, and absence of saved outputs without executing cells.

To validate the clean notebook without making API requests:

```bash
python validate_notebook.py
```

The notebook was copied through PyCharm's notebook model; the original working notebook and its outputs are not modified. No API keys are included.

## Web app and Vercel deployment

`app.py` serves the FastAPI endpoints and the interface in `public/`. The runtime workflow lives in `planner/`; `vercel.json` configures deployment. Deployment has not been verified.

Install `requirements.txt`, copy `.env.example` to `.env`, and supply these server-side variables:

- `OPENAI_API_KEY`: model access, created at https://platform.openai.com/api-keys.
- `TAVILY_API_KEY`: web research, available at https://app.tavily.com.
- `SERPAPI_API_KEY`: Google Flights and Hotels searches, available at https://serpapi.com/manage-api-key.
- `APP_ACCESS_TOKEN`: your own private app password. Generate a random value with `openssl rand -hex 32`, save it in `.env`, and enter it under Settings → Invitation code in the web interface. It is required on Vercel and optional locally.

Run locally with `uvicorn app:app --reload`, then open http://127.0.0.1:8000.

For Vercel, import this repository and set all four variables in the project's environment settings before deploying. Apply them to each environment where live planning should work. Redeploy after changing deployment variables. Keep provider keys on the server and never commit `.env` or paste provider keys into the web interface.

## Public beta security controls

The web runtime uses layered controls, not a claim that prompt injection is solved:

- `omni-moderation-latest` checks each accepted input and the final answer. Moderation failures, missing results, and flagged content stop the request. Moderation checks harmful-content categories; it is not a prompt-injection detector.
- Local Unicode-normalized regex checks reject common instruction overrides, prompt/secret extraction, role markers, code-execution instructions, credentials, emails, phone numbers, labeled IDs, bank accounts, and Luhn-valid payment-card numbers. Final answers have configured secrets and patterned private data removed before moderation and delivery. Pattern matching cannot recognize every PII format or obfuscated attack.
- There is **no model-based NER** and no extra generative guardrail-model call. The existing routing model also classifies travel scope and rejects unrelated or malicious requests before tool dispatch. A model classifier remains probabilistic.
- Every agent and the final synthesizer receive the same security policy. Retrieved content is treated as data; recognizable instruction injections are omitted from tool observations. Public responses exclude raw agent messages, internal findings, and prompts.
- The web runtime no longer uses the Deep Agents default tool harness. Research has only a bounded web-search loop; there is no model-accessible filesystem, shell, arbitrary URL fetcher, payment, booking, or messaging tool. Provider endpoints are fixed in code.
- Each request is capped at 16 model invocations, eight paid search calls, bounded tool batches, 2,000 output tokens per planner-model invocation, 50,000 characters of model input, and 180 seconds total. SDK retries are disabled for planner models and moderation. These are operation limits, **not an exact dollar budget**. Provider usage/billing limits remain important.
- Request bodies are limited to 24 KiB, query text to 6,000 characters, and unexpected JSON fields are rejected. Cross-site browser submissions are rejected. The access key is checked before reading the body or starting paid work.
- Vercel sends a restrictive Content Security Policy, framing restrictions, and disabled camera/microphone/geolocation/payment permissions. Model Markdown is rendered as text and a small element allowlist rather than executed as HTML.

### Usage protection without an extra service

No Redis database or additional service account is required. Each worker keeps best-effort in-memory limits: **three plans per fixed UTC minute, 30 per UTC day, and two concurrent plans**. Accepted requests consume allowance even if moderation declines them or a provider fails. Active leases expire after crashes; completed requests release them.

These counters are **not app-wide**: they reset on restarts and are independent across Vercel instances. They do not establish a reliable daily budget or stop a determined token holder from exploiting serverless scaling. The access key, per-request call/token/time limits, input/output checks, and restricted tools remain enforced independently of these counters. Keep the beta invitation-only and configure provider-side usage controls as appropriate.

Set `PLANNING_ENABLED=false` and redeploy for an emergency pause. A production access token must contain at least 32 characters. A shared token can be redistributed; use individual sign-in and shared storage for durable per-user quotas if broader public access is needed later.

Run the offline safety suite with `python -m pytest tests -q` after installing pytest in the development environment. Tests stub external services and exercise enforcement and failure paths without charging providers. They are not a live adversarial evaluation of the models.

## LangSmith observability

The browser displays progress and the final travel plan. Tool calls, agent messages, and execution traces are never presented in the UI or included in downloaded plans. Observability is server-side through LangSmith.

Add the following to your local `.env` and the Vercel project's environment settings, then restart locally or redeploy:

```dotenv
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=travel-planner
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

For an EU LangSmith account, use `https://eu.api.smith.langchain.com`. If your key is associated with multiple workspaces, also set `LANGSMITH_WORKSPACE_ID`. Keep the API key server-side. LangSmith is optional: missing credentials or telemetry failures do not disable planning.

The private project contains a parent `travel_plan_request` trace with nested graph nodes, model invocations, tools, and input/output guardrail checks. Safe metadata records a generated request ID, deployment revision/environment, request outcome, model-call count, and search-call count. LangChain model usage metadata is retained for LangSmith's token and cost reporting.

The trace client always hides inputs and outputs. A final batch filter also removes trace events, attachments, serialized prompts, runtime details, unexpected metadata, and raw exception text. Known credentials and patterned PII are masked in remaining strings. No user headers or access keys are used as tracing metadata. This deliberately limits content debugging to protect travelers; inspect workflow structure and safe operational metadata instead. Do not add raw prompts or search results to metadata.

Trace batches are flushed with a short timeout before the serverless request finishes. Delivery remains best-effort during platform termination or LangSmith outages. Auth/body-validation/rate-limit rejections happen before the workflow trace starts; these remain HTTP errors rather than billable workflow traces. Check your LangSmith plan's trace allowance before inviting a wider audience.
