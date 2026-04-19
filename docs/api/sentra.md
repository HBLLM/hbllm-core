# Sentra Agent Framework

> Build autonomous AI agents in minutes, powered by [HBLLM Core](../index.md).

## Quick Start

### 1. Install

```bash
pip install sentra
# or for development:
pip install -e ./sentra[dev]
```

### 2. Initialize Project

```bash
sentra init
```

This interactive wizard will:
- Ask you to choose a provider (OpenAI, Anthropic, Ollama, or Local HBLLM)
- Set up API keys
- Create `~/.sentra/config.yaml` (global) and `agent.yaml` (project)
- Create a workspace directory

### 3. Run Your Agent

```bash
sentra run assistant
```

---

## Configuration

### Global Config (`~/.sentra/config.yaml`)

```yaml
provider:
  name: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}     # References environment variable
```

### Project Config (`agent.yaml`)

```yaml
version: "1"
agents:
  researcher:
    tenant_id: team_alpha
    workspace_path: ./researcher_workspace
    system_prompt: "You are a research assistant."
    tools:
      - web_search
      - calculator
  coder:
    tenant_id: team_alpha
    provider:
      name: anthropic
      model: claude-sonnet-4-20250514
```

### Supported Providers

| Provider | Config | Auth |
|----------|--------|------|
| **OpenAI** | `name: openai, model: gpt-4o-mini` | `OPENAI_API_KEY` |
| **Anthropic** | `name: anthropic, model: claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| **Ollama** | `name: openai, base_url: http://localhost:11434/v1` | None needed |
| **Local HBLLM** | `name: local, model: hbllm-local` | None needed |

### CLI Commands

```bash
sentra init                          # Setup wizard
sentra config show                   # View current config
sentra config set-key openai sk-...  # Set API key
sentra run [agent_name]              # Run agent REPL
sentra list                          # List configured agents
```

---

## Python API

### Basic Agent

```python
from sentra import SentraAgent, tool

@tool(description="Search the web")
def web_search(query: str) -> str:
    return f"Results for: {query}"

agent = SentraAgent(
    name="researcher",
    tenant_id="default",
    workspace_path="./workspace",
)
await agent.start()

# Execute a task
result = await agent.execute_task({"text": "What is quantum computing?"})
print(result)

await agent.stop()
```

### Config-Driven Setup

```python
from sentra import load_config, build_provider_from_config, SentraAgent

config = load_config(project_path=".")
agent_cfg = config.agents["researcher"]
llm = build_provider_from_config(agent_cfg.provider)

agent = SentraAgent(
    name=agent_cfg.name,
    tenant_id=agent_cfg.tenant_id,
    workspace_path=agent_cfg.workspace_path,
    llm=llm,
)
```

### Streaming

```python
async for chunk in agent.stream_task({"text": "Explain AI in detail"}):
    print(chunk, end="", flush=True)
```

### Memory

```python
await agent.remember("The capital of France is Paris")
memories = await agent.recall("capital of France")
```

### Task Planning (Graph-of-Thoughts)

```python
plan = await agent.planner.plan("Design a microservices architecture")
# Returns a DAG of subtasks via MCTS-guided reasoning
```

### Governance

```python
check = await agent.governance.validate("Generated content to review")
# Returns policy compliance result with violation log
```

### Self-Improvement

```python
eval_result = await agent.improvement.evaluate("task input", "task output")
insights = await agent.improvement.reflect("Session summary")
```

---

## Multi-Agent Swarm

```python
from sentra import SentraOrchestrator, SentraAgent

orchestrator = SentraOrchestrator(tenant_id="team_alpha")

researcher = SentraAgent(name="researcher", tenant_id="team_alpha", workspace_path="./ws_r")
writer = SentraAgent(name="writer", tenant_id="team_alpha", workspace_path="./ws_w")

orchestrator.register_agent(researcher)
orchestrator.register_agent(writer)

await orchestrator.start()
result = await orchestrator.submit_task("researcher", {"text": "Find trending AI papers"})
```

---

## MCP Server

Expose your agents as MCP-compatible tools for external LLM integration:

```python
from sentra import SentraMCPServer, SentraAgent

server = SentraMCPServer(name="my-agents", version="1.0.0")
agent = SentraAgent(name="helper", tenant_id="t1", workspace_path="./ws")
server.register_agent(agent)

# Handles MCP protocol:
tools = await server.handle_list_tools()
result = await server.handle_tool_call("execute_task", {"text": "Hello"})
```

---

## Architecture

```
SentraAgent (8 nodes)
├── WorkspaceNode      — Shared blackboard memory
├── RouterNode         — Intent classification + domain routing
├── ToolRouterNode     — Tool dispatch via @tool decorator
├── MemoryNode         — Episodic + semantic memory (SQLite)
├── PlannerNode        — Graph-of-Thoughts MCTS planner
├── SentinelNode       — Guardrail enforcement (policies)
├── EvaluationNode     — Output quality scoring
└── ReflectionNode     — Self-improvement via reflection
```

### Module Reference

| Module | Class | Purpose |
|--------|-------|---------|
| `agent.py` | `SentraAgent` | Core agent wrapper |
| `tools.py` | `@tool`, `SentraToolNode` | Tool registration |
| `memory.py` | `SentraMemoryClient` | Episodic memory |
| `swarm.py` | `SentraOrchestrator` | Multi-agent coordination |
| `planner.py` | `SentraPlanner` | Task decomposition |
| `governance.py` | `SentraGovernance` | Policy enforcement |
| `self_improvement.py` | `SentraSelfImprovement` | Autonomous learning |
| `config.py` | `SentraConfig`, `load_config` | YAML configuration |
| `cli.py` | `main()` | Command-line interface |
| `streaming.py` | `SentraStream` | Token streaming |
| `mcp_server.py` | `SentraMCPServer` | MCP protocol server |
| `cloud_bridge.py` | `SentraCloudBridge` | Enterprise cloud (opt-in) |
| `benchmarks.py` | `run_all_benchmarks()` | Performance suite |
