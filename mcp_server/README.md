# connect4 MCP server (local, git-ignored)

Install once: `.venv/bin/pip install -r mcp_server/requirements.txt`
Run (stdio): `.venv/bin/python mcp_server/server.py`

Claude Code picks it up from `.mcp.json` in the repo root.
Claude Desktop: add the same `connect4` block from `.mcp.json` to
`~/.config/Claude/claude_desktop_config.json` under `mcpServers`.
Debug: `npx @modelcontextprotocol/inspector .venv/bin/python mcp_server/server.py`

Tools: list_runs, run_metrics, run_eval, analyze_position, solve_position,
minimax_move, bench_accuracy, arena_summary, head_to_head, list_results,
read_result, list_docs, read_doc. Resource: connect4://research-notes.
Moves are 1-indexed column strings ("4453"); output columns are 0-indexed.
