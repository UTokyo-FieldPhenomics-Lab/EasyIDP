# EasyIDP Roadmap And Branching Rules

When planning or implementing EasyIDP architecture work, use this roadmap as the current project direction. Do not treat older archived plans as binding if they conflict with this rule.

## Version Roadmap

- v2.1: Adopt the new architecture first. Partial old API breakage is acceptable if it makes the public interfaces cleaner and easier to maintain.
- v2.2: Build enhancements for unresolved small features from issues on top of the new architecture.
- v3.0: Add MCP and skills support.
- v3.1: Gradually implement compatibility with more photogrammetry/reconstruction software.

## Branching Strategy

- `main`: mainstream stable release branch.
- `dev`: active development branch. Implement new architecture work here first, then merge to `main` after validation.
- Remove legacy maintenance branches such as `v2.0`, `v1.0`, and similar old version branches when appropriate.

## Architecture Bias

- Prioritize new architecture clarity over strict backward compatibility.
- Develop directly on `dev` unless the user asks for a separate worktree or feature branch.
- Prefer clean, stable module boundaries that can support future features without accumulating compatibility shims.
