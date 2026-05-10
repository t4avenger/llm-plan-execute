# 0005 Report exact usage when available and estimates otherwise

- Status: Accepted
- Date: 2026-05-10

## Context

The final report should include token usage and possible cost, but provider CLIs do not expose usage data consistently. Hiding unavailable usage would make reports incomplete, while presenting estimates as exact would be misleading.

## Decision

Report exact token usage and cost when a provider exposes it. Otherwise estimate tokens and cost locally, and label the confidence as estimated in both JSON artifacts and Markdown reports.

## Consequences

Reports are useful from v1 while staying honest about measurement quality. Cost estimates are approximate until adapters can parse exact provider usage. Reports must preserve exact/estimated confidence per model call.
