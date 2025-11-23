# Versioning and Release Workflow

This project uses **Semantic Versioning** automated by **Conventional Commits** and **GitHub Actions**.

## How it Works

The release pipeline is triggered automatically when changes are pushed to the `main` branch. The system analyzes your commit messages to determine the next version number.

### 1. Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Your commit messages determine the version bump:

| Commit Type | Impact | Version Change | Example |
| :--- | :--- | :--- | :--- |
| `fix:` | **Patch** | `0.0.1` -> `0.0.2` | `fix: correct retry logic` |
| `feat:` | **Minor** | `0.0.1` -> `0.1.0` | `feat: add user profile api` |
| `refactor!:` | **Major** | `0.0.1` -> `1.0.0` | `refactor!: drop support for python 3.8` |
| `BREAKING CHANGE` | **Major** | `0.0.1` -> `1.0.0` | (Footer in any commit) |
| `docs:`, `chore:`, etc. | **None** | No release | `docs: update readme` |

> **Note**: A **Major** version bump always resets Minor and Patch versions to `0`.
> Example: `1.2.3` + Major -> `2.0.0` (NOT `2.2.3` or `2.0.3`).

### 2. The Release Pipeline

When you push to `main`:

1. **GitHub Action Triggered**: The `.github/workflows/release.yml` workflow starts.
2. **Analyze Commits**: `python-semantic-release` scans commits since the last tag.
3. **Calculate Version**: It determines the next version based on the highest impact commit.
4. **Update Files**:
   - Updates `version` in `pyproject.toml`.
   - (Optional) Updates `CHANGELOG.md`.
5. **Publish**:
   - Creates a new Git Tag (e.g., `v1.0.0`).
   - Creates a GitHub Release with release notes.
   - (Configured) Publishes to PyPI or other registries if enabled.

### 3. Manual Workflow

You generally do **not** need to manually bump versions. Just focus on writing descriptive commit messages.

#### Example Workflow
1. You work on a new feature.
2. Commit: `feat: add vector database support`.
3. Push to `main`.
4. **Result**: CI sees `feat`, bumps version from `0.1.0` to `0.2.0`, and releases it.
