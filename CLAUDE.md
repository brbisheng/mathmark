# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context
- Mintlify documentation project (repo is empty aside from this file; no build/lint/test commands are defined locally).
- Content is MDX with YAML frontmatter; navigation/theme/settings live in `docs.json`; pages use Mintlify components.

## Working relationship
- Push back on ideas — cite sources and explain reasoning.
- Ask for clarification rather than assume.
- Never lie, guess, or make up information.

## Adding your first page
- Search the tree for an existing page on the topic before creating a new one.
- When you add `docs.json`, model `navigation` on Mintlify's recommended shape — don't invent a custom schema.
- For the first MDX file, copy the Mintlify starter frontmatter and set only `title` and `description`.
- Make the smallest reasonable change — one page, one config block, no speculative sub-pages.

## Frontmatter (required)
- `title`: clear, descriptive.
- `description`: one-sentence summary.

## Writing standards
- Second-person voice.
- Prerequisites up front on procedural pages.
- Test every code block; tag every code block with a language.
- Alt text on every image.
- Relative paths for internal links.
- Match style and structure of existing pages.

## Do not
- Skip frontmatter on any MDX file.
- Use absolute URLs for internal links.
- Include untested code examples.
- Assume intent or invent API behavior — ask.

## Git workflow
- No hooks are configured today; if/when they are, do not bypass them.
- Ask how to handle uncommitted changes before starting.
- Create a new branch when no clear branch exists.
- Commit frequently.
