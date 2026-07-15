# Model Workflow Experience

The user of this repository is an analyst making a retention decision, not merely a person waiting for Python to finish. The workflow therefore makes progress, evidence, artifacts, and human review explicit.

## Primary journey

1. Discover and validate a compatible dataset.
2. Compare candidate models with visible progress.
3. Refit the selected model and report its threshold.
4. Review the decision summary and generated artifacts.
5. Inspect false negatives before operational use.

## Output contract

- Progress lines use numbered stages so long-running work never looks frozen.
- Model comparison prints the current candidate and total candidate count.
- The final summary groups performance metrics separately from artifacts.
- Prediction output shows probabilities and the human-readable status, not engineered features.
- Failures must name the missing column or incompatible file and explain the next corrective action.

## Reporting rules

- Never encode model quality with color alone; keep metric names and values visible.
- Any chart built from the CSV outputs needs a title, axis labels, legend, and a text or table equivalent.
- Round display values consistently while preserving raw values in exported files.
- Treat probability as evidence, not certainty. Surface the selected threshold beside every decision view.
- Keep false-negative review visible because missed churners are an important business risk.

## Future interface states

`dataset needed` -> `validating` -> `training` -> `evaluating` -> `ready for review` -> `approved for experiment`

An operational dashboard should never jump directly from a score to an automated customer action. It should preserve explanation, reviewer identity, threshold version, and model version.
