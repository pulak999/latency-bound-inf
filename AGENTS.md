# Agent instructions

## GitHub issues

Use [the research issue template](.github/ISSUE_TEMPLATE/experiment-task.md)
when you create or rewrite a GitHub issue for this repository.

Write the issue in ASD-STE100 Simplified Technical English. Use active voice.
Use one command per instruction. Use a maximum of 20 words in an instruction
and 25 words in a descriptive sentence. Use the same term for the same thing.
Use the ASD-STE100 Issue 9 dictionary. Verify the approved meaning and part
of speech. Apply the rules for technical nouns and verbs. Define project
terms or link to the glossary in issue #20.

Keep the three headings in the template. Replace the prompts with issue text.
Use subheadings for the extra detail in roadmaps and specifications.
Keep the experiment tracker layout in issue #20.
For an experiment, give the prediction and the check in Expected outcomes.
The measured result can differ from the prediction.

## Issue closure

Use the current Expected outcomes in the issue as the acceptance criteria.
Keep the agreed criteria when a result differs from a prediction.

Before you close an issue, post one [results comment](.github/issue-results-comment.md).
Include one row for each expected outcome, in issue order. Give the required
outcome, the observation, the evidence link, and the remaining work.
Write “None” when the evidence covers the full outcome.

Put the decisive measurement or test output in the row. Include units and
counts where they apply. Link the tested commit and the raw records.
Give the exact command that reproduces the result. Use permanent links
that another maintainer can open. Identify any unrecorded code changes.

Include a run setting when it affects the result. For deadline results,
give the tenant, time origin, deadline, and number of scheduled frames.
Include frames without answers in the miss count. For a comparison across
runs, identify the baseline and report the variation across repeats.

Use a device reading for a claim about hardware state. Use measured run
data for a claim about performance. A result can answer a research
question even when it differs from the prediction.

Close the issue when the evidence covers every expected outcome.
If evidence is missing, keep the issue open. Name the smallest remaining
check. If all outcomes for one milestone are complete, record that
milestone. Keep the roadmap open while other outcomes remain.

Publish the results comment before any action that closes the issue.
