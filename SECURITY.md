# Security policy

This repository is a research model, not a production banking service.

## Data handling

- Do not commit customer CSVs, account identifiers, exported predictions, credentials, or Kaggle tokens.
- Keep training data in an access-controlled location and record only a non-sensitive dataset release or checksum.
- Treat generated prediction files as sensitive; remove them from shared artifacts unless they are anonymized and explicitly approved.
- The model output must not be used as an automatic approval, denial, pricing, or customer-contact decision without qualified human review and documented validation.

## Reporting

Report suspected credential exposure, unsafe deserialization, data leakage, or exploitable code privately to the repository owner. Include reproduction steps without attaching customer data. Do not open a public issue for an active vulnerability.

The project uses serialized model artifacts; load only artifacts from a trusted, integrity-checked source.
