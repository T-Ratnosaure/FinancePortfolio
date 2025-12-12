# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
GO THROUGH THIS FILE WITH SERIOUS. RESPECT IT ALL EVERY TIME.

---

## ⚠️ MANDATORY WORKFLOW - READ FIRST ⚠️

**These rules are NON-NEGOTIABLE. Failure to follow them is unacceptable.**

### 1. ALWAYS USE AGENTS
- **NEVER** try to do everything yourself
- **ALWAYS** use the Task tool to launch specialized agents for their domains
- Launch multiple agents in parallel when tasks are independent
- Use managers (Jacques, Jean-David, Jean-Yves) for coordination

### 2. AFTER EVERY MAJOR DEVELOPMENT
After completing any significant feature, fix, or change:
1. Create a new feature branch: `git checkout -b feat/description` or `fix/description`
2. Commit changes with conventional commit messages
3. Push the branch: `git push -u origin <branch-name>`
4. **Launch review agents** before creating PR:
   - `it-core-clovis` - Git workflow & code quality review
   - `lamine-deployment-expert` - CI/CD & TDD review
   - `legal-compliance-reviewer` or `legal-team-lead` - Compliance review
5. Address all blocking feedback from reviewers

### 3. CREATE PR THROUGH GITHUB CLI
GitHub CLI (`gh`) is installed at: `"/c/Program Files/GitHub CLI/gh.exe"`
```bash
"/c/Program Files/GitHub CLI/gh.exe" pr create --title "type(scope): description" --body "..."
"/c/Program Files/GitHub CLI/gh.exe" pr list
"/c/Program Files/GitHub CLI/gh.exe" pr checks <number>
"/c/Program Files/GitHub CLI/gh.exe" pr view <number>
```

### 4. MERGE ONLY IF CI PASSES
- **NEVER** merge a PR if CI pipeline fails
- Check CI status: `"/c/Program Files/GitHub CLI/gh.exe" pr checks <number>`
- Wait for CI to complete before merging
- If CI fails, fix the issues and push again
- Merge command: `"/c/Program Files/GitHub CLI/gh.exe" pr merge <number> --squash --delete-branch`

### 5. WRITE SPRINT REVIEW DOCUMENTATION
After each sprint/major milestone, create/update `docs/reviews/sprint-X-review.md`:
- What was accomplished
- What's planned for next sprint
- Compliance agent feedback summary
- Any risks or concerns raised
- Lessons learned

---

## Project Overview

FinancePortfolio is a Python 3.12 application managed with UV (fast Python package manager).

## Development Commands

### Package Management
- Install dependencies: `uv sync`
- Add a new dependency: `uv add <package-name>`
- Add a dev dependency: `uv add --dev <package-name>`
- FORBIDDEN: `uv pip install`, `@latest` syntax
- ONLY use uv, NEVER pip

### Running the Application
- Run main script: `python main.py`
- Or with UV: `uv run python main.py`

## Project Structure

- `main.py` - Application entry point with basic setup
- `pyproject.toml` - Project configuration and dependencies
- `.python-version` - Python version (3.12)

## Core Development Rules

### Code Quality

- Type hints required for all code
- use pyrefly for type checking
  - run 'pyrefly init' to start
  - run 'pyrefly check' after every change and fix resultings errors
- Public APIs must have docstrings
- Functions must be focused and small
- Follow existing patterns exactly
- Line length: 88 chars maximum

### Testing Requirements
   - Framework: `uv run pytest`
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

### Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## System Architecture

- use pydantic and langchain

## Multi-Agent Architecture

This project uses a multi-agent system. **Subagents MUST be called to work on tasks.**

### When to Use Subagents
- **ALWAYS** use the Task tool to launch specialized agents for their domains
- **DO NOT** try to do everything yourself - delegate to the appropriate expert
- Launch multiple agents in parallel when tasks are independent

### Available Teams and When to Call Them

| Team | Manager | When to Call |
|------|---------|--------------|
| **Research** | Jean-Yves | Portfolio analysis, ML models, market signals, quantitative analysis |
| **Data** | Florian | Data pipelines, ETL, data quality, API integrations |
| **Legal** | Marc | Compliance, tax optimization, regulatory questions |
| **IT-Core** | Jean-David | Code quality, CI/CD, git workflow, security |
| **Risk** | Nicolas | VaR, position limits, drawdown analysis |
| **Execution** | Helena | Trading execution, backtesting |

### Key Agents by Specialty

| Agent | Specialty | Use When |
|-------|-----------|----------|
| `research-remy-stocks` | Equity quant, stochastic calculus | ETF analysis, volatility, options |
| `iacopo-macro-futures-analyst` | Macro, rates, FX | Economic indicators, regime analysis |
| `alexios-ml-predictor` | ML model design | Building predictors, feature engineering |
| `antoine-nlp-expert` | NLP, sentiment | Text analysis, LLM integration |
| `data-engineer` | Data pipelines | Data sourcing, ETL design |
| `french-tax-optimizer` | French tax law | PEA optimization, tax strategy |
| `lamine-deployment-expert` | CI/CD, TDD | Deployment, testing infrastructure |
| `quality-control-enforcer` | Code quality | Review implementations |

### Usage Pattern

```
1. Identify what expertise is needed
2. Launch appropriate agent(s) with Task tool
3. Run multiple agents in parallel when possible
4. Synthesize outputs from multiple agents
5. Use managers (Jacques, Jean-Yves, etc.) for coordination
```

### Example

```
User asks about portfolio optimization:
1. Launch research-remy-stocks for ETF analysis
2. Launch iacopo-macro-futures-analyst for regime detection
3. Launch french-tax-optimizer for PEA considerations
4. Synthesize all outputs into recommendation
```

## Documentation

### README.md
- **ALWAYS update the README.md** when completing a phase or adding major functionality
- Keep it synchronized with the current state of the project
- Include:
  - Project overview and goals
  - Implementation plan summary (phase progress)
  - Completed features and accomplishments
  - Setup instructions
  - Tech stack and dependencies
  - Usage examples (when applicable)

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

## Git Workflow

- Always pull when comming back to the code
- Always use feature branches; do not commit directly to `main`
  - Name branches descriptively: `fix/XXX`, `feat/XXXX`, `chore/ruff-fixes`
  - Keep one logical change per branch to simplify review and rollback
- Create pull requests for all changes
  - Open a draft PR early for visibility; convert to ready when complete
  - Ensure tests pass locally before marking ready for review
  - Use PRs to trigger CI/CD and enable async reviews
- Link issues
  - Before starting, reference an existing issue or create one
  - Use commit/PR messages like `Fixes #123` for auto-linking and closure
- Commit practices
  - Make atomic commits (one logical change per commit)
  - Prefer conventional commit style: `type(scope): short description`
    - Examples: `feat(eval): group OBS logs per test`, `fix(cli): handle missing API key`
  - Squash only when merging to `main`; keep granular history on the feature branch
- Practical workflow
  1. Create or reference an issue
  2. `git checkout -b feat/issue-123-description`
  3. Commit in small, logical increments
  4. `git push` and open a draft PR early
  5. Convert to ready PR when functionally complete and tests pass
  6. Merge after reviews and checks pass

## Python Tools

- use context7 mcp to check details of libraries

## Code Formatting

1. isort (AUTHORITATIVE for import sorting)
   - Check: `uv run isort --check-only --diff .`
   - Fix: `uv run isort .`
   - **isort always prevails** over other tools for import ordering
   - Configuration in `pyproject.toml` uses "black" profile
   - Ruff's import sorting rules (I) are disabled - isort is the authority
   - Run isort BEFORE ruff format

2. Ruff
   - Format: `uv run ruff format .`
   - Check: `uv run ruff check .`
   - Fix: `uv run ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Unused imports
   - Note: Import sorting (I001) is handled by isort, not ruff
   - Line wrapping:
     - Strings: use parentheses
     - Function calls: multi-line with proper indent
     - Imports: split into multiple lines

3. Type Checking
   - run `pyrefly init` to start
   - run `pyrefly check` after every change and fix resultings errors
   - Requirements:
     - Explicit None checks for Optional
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass

4. Security Scanning (Bandit)
   - Check: `uv run bandit -c pyproject.toml -r .`
   - Configuration in `pyproject.toml` (excludes tests, .venv)
   - Severity levels: low, medium, high
   - CI runs with `--severity-level medium` (medium+ issues fail)
   - Ruff also has S rules (flake8-bandit) for fast feedback
   - Bandit provides deeper analysis as second layer

5. Complexity Analysis (Xenon)
   - Check: `uv run xenon --max-absolute B --max-modules A --max-average A . --exclude ".venv,venv"`
   - Grades: A (best) to F (worst)
   - Thresholds:
     - `--max-absolute B`: No single block worse than B
     - `--max-modules A`: Module average must be A
     - `--max-average A`: Overall average must be A
   - Ruff C901 rule also enforces max-complexity of 10

6. Pre-commit Hooks (Automated Quality Gates)
   - Install: `uv run pre-commit install` (one-time setup)
   - Manual run: `uv run pre-commit run --all-files`
   - Auto-runs on `git commit` to prevent broken code from being committed
   - Hook execution order:
     1. File hygiene (trailing whitespace, EOF, YAML/TOML syntax)
     2. isort (import sorting)
     3. ruff format (code formatting)
     4. ruff check (linting)
     5. bandit (security scanning)
     6. pyrefly (type checking)
   - Configuration in `.pre-commit-config.yaml`
   - All hooks use `uv run` for consistency with CI and local development
   - Hooks will FAIL the commit if any check fails
   - To bypass hooks (NOT recommended): `git commit --no-verify`
   - Best practice: Fix issues instead of bypassing hooks

## Error Resolution

1. CI Failures
   - Fix order:
     1. Import sorting (isort)
     2. Formatting (ruff format)
     3. Linting (ruff check)
     4. Security issues (bandit)
     5. Complexity issues (xenon)
     6. Type errors
   - Type errors:
     - Get full line context
     - Check Optional types
     - Add type narrowing
     - Verify function signatures

2. Common Issues
   - Line length:
     - Break strings with parentheses
     - Multi-line function calls
     - Split imports
   - Types:
     - Add None checks
     - Narrow string types
     - Match existing patterns

3. Best Practices
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Follow existing patterns
   - Document public APIs
   - Test thoroughly

---

## ⛔ LESSONS LEARNED - NEVER DO THIS AGAIN ⛔

**These are mistakes made during development. Learn from them and NEVER repeat.**

### Security Anti-Patterns (Sprint 4 Discoveries)

1. **NEVER use `pickle` for serialization**
   - Pickle allows arbitrary code execution on load (CVSS 9.8 - CRITICAL)
   - Use `joblib` for ML models + JSON for config data
   - If you must serialize complex objects, use Protocol Buffers or safe alternatives
   - **Bad**: `pickle.load(open(file, 'rb'))`
   - **Good**: `joblib.load(file)` with JSON metadata

2. **NEVER forget `.env` in `.gitignore`**
   - API keys, secrets, credentials WILL be exposed
   - Add these patterns to `.gitignore` IMMEDIATELY on project creation:
     ```
     .env
     .env.*
     *.key
     *.pem
     secrets/
     credentials/
     ```

3. **NEVER trust user-provided file paths**
   - Always validate paths to prevent directory traversal attacks
   - Block system directories (`/etc`, `/usr`, `C:\Windows`, etc.)
   - Validate file extensions for sensitive operations

### Code Quality Anti-Patterns

4. **NEVER reference non-existent code in examples**
   - Sprint 4 had broken examples referencing `ETFSymbol.SPY` and `ETFSymbol.AGG` which didn't exist
   - Examples should be TESTED and WORKING
   - Run example scripts as part of CI or manual verification

5. **NEVER skip formatting before pushing**
   - Always run `uv run ruff format .` before committing
   - CI will fail on formatting issues
   - Amending commits wastes time and clutters history

6. **NEVER block `/tmp` in path validation for libraries**
   - pytest and other tools use `/tmp` for temporary files on Linux
   - CI runs on Linux and will fail if `/tmp` is blocked
   - Use environment detection (`PYTEST_CURRENT_TEST`) for test-aware validation

### Mathematical Anti-Patterns

7. **NEVER implement financial formulas without verification**
   - Sprint 4 discovered wrong Sortino ratio formula
   - **Wrong**: `std(negative_returns_only)`
   - **Correct**: `sqrt(mean((returns - target)^2 where returns < target))`
   - Always verify formulas against authoritative sources (CFA curriculum, academic papers)

### Process Anti-Patterns

8. **NEVER skip multi-team reviews**
   - Sprint 3 was merged without thorough security/quality review
   - Result: Critical vulnerabilities discovered post-merge
   - Always launch review agents BEFORE creating PR:
     - `cybersecurity-expert-maxime` for security
     - `quality-control-enforcer` for code quality
     - `research-remy-stocks` or `portfolio-manager-jean-yves` for financial math

9. **NEVER let supporting code quality slip**
   - Core modules (`src/`) were excellent
   - Supporting code (`examples/`, `main.py`) was broken
   - ALL code should meet the same quality standards

### CI/CD Anti-Patterns

10. **NEVER assume local tests = CI success**
    - Windows paths differ from Linux paths
    - Temp directories are in different locations
    - Always check CI logs when it fails, don't guess
