# Ares — Projects4Me AI service (Python 3.11, FastAPI, Anthropic SDK)

## Run
- Run command and test approach: workspace skills `p4m-stack` and `run-tests` (no tests yet; never call the real API in tests).

## Architecture
- `routers/planissue.py` (`POST /planissue/`, prompts) -> `services/planissue_service.py` -> `factories/ai_service_factory.py` -> `adapters/anthropic_adapter.py`.
- Only caller: Gaia `core/services/LLM.php` (POSTs `{issueDetails:{projectName, issueSubject, issueDescription}}`).
- New providers: subclass `adapters/base_ai_adapter.py` and register in the factory.

## Rules
- Use Pydantic models for request/response; return parsed JSON, not a JSON string (coordinate the change with Gaia).
- Use the async Anthropic client inside async routes.
- Model name and limits come from env, not code.
- Treat issue text as untrusted input in prompts.

## House rules

These house rules assume the repo is checked out inside the **Pr4M** workspace (https://github.com/projects4me/Pr4M), which provides `dev-tools/` (the `gaia` MCP server, the shared Agent Odysseus identity, the `ai` workflow CLI, Talos, the `installer`) and the workspace skills and agents in `.claude/`.

### Workflow
- **Issue first.** Before starting any work, find or create its issue in Projects4Me: search with `gaia_list_issues`, create with `gaia_create_issue` (the `gaia` MCP server in `dev-tools/gaia-mcp`), or use the Projects4Me UI. Ares work goes in project **A Ares** (short code `aa`). Use its issue number in the branch name (`claude-issue-NNNN`). Move the issue's status as work progresses (`in_progress` → `in_review` → `done`).
- **Milestone, time and notes.** Every issue goes in the project's **Claude-v1** milestone (the `gaia` MCP server does this by default). While working, log the time spent on the issue (`gaia_log_time`, with a short description of what it was spent on) and record the details in its thread (`gaia_add_comment`): what was done, findings, decisions, files changed and what's left. Do both before marking the issue `in_review`.
- **Branch, PR, merge.** Move the issue to `in_progress`, then branch `claude-issue-NNNN` off an up-to-date `dev`. Run the full test suite before opening the PR. Push the branch to the `odysseus` remote and open a PR to `dev` with `odysseus gh pr create`. **Never merge**: the user merges. After the user merges, rebase local `dev` onto `origin/dev` before starting the next issue. Leave uncommitted files you didn't create for this issue alone.
- **Identity: Agent Odysseus.** Commit, push and open PRs only as Agent Odysseus, through `dev-tools/odysseus/odysseus git …` / `odysseus gh …` (see `dev-tools/odysseus/README.md`). Push only to the `odysseus` remote, never to `origin`. Never change git or gh configuration (global or repo) and never use your personal git/GitHub credentials for agent work; if the agent identity isn't available, stop and ask.
- One issue per PR; log anything else you find in the Projects4Me tracker instead of fixing it in the same PR.
- MVP first: items in the Engineering Guide's pre-production table are out of scope unless the issue asks for them.
- Never commit, push, merge, force-push, tag or deploy without explicit approval. Never commit to `master` or `qa`.
- When docs and code disagree, the code wins; flag the doc in your summary.
- Acceptance criteria unclear? Ask instead of guessing, and write down assumptions in the PR.

### Boundaries
- Stateless: no database, no stored data. Gaia `core/services/LLM.php` is the only caller.
- Request/response changes ship as linked PRs with Gaia and merge together.
- Failures must be visible: return a clear error status and message, never an empty or fake plan.

### Security
- Never commit `ANTHROPIC_API_KEY` or other secrets; never log keys or full prompts containing user text.
- No live Anthropic calls in tests or CI; use a fake adapter.
- Keep debugpy off outside local dev.

### Design (Fowler's layering + SOLID)
- Follow the layering and separation of concerns from Martin Fowler's *Patterns of Enterprise Application Architecture*: presentation (`routers/`: request/response models and HTTP only) → service layer (`services/`: the use case) → gateways (`adapters/` behind `base_ai_adapter.py`, chosen by `factories/`). Dependencies point down only; routers hold no prompt or provider logic, and services don't import a concrete provider.
- SOLID, especially **Single Responsibility**: one reason to change per module, class or function. New providers extend the adapter base (open/closed); depend on the base adapter, not a concrete one.
- Before adding a module, class or function, name its single responsibility; if it needs "and", split it.

### Quality
- "Done" means tests and lint were run and passed. Report failures as they are, and say what wasn't run.
- A bug fix starts with a test that fails.
- No new dependencies and no framework or runtime upgrades without approval.

### PR description
What · Why (issue link) · How tested · Contract impact (events, envelope, Ares format) · Migration impact · Out of scope / follow-ups.
