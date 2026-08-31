# Contributing

Create an isolated Python environment and install the project requirements:

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

Changes to features, thresholds, or model selection must include a test or an evaluation note. Keep the held-out test split untouched during model selection and inspect false negatives before describing a result.

Do not commit customer records, Kaggle credentials, generated predictions, or serialized artifacts from untrusted sources.
