# Synthetic AgentCore A/B test fixtures

Hand-built `GetABTest` and `GetConfigurationBundleVersion` responses for the A/B audit's tests, written
to the installed botocore models (1.43.106) and validated by `Stubber` when the tests run. The account
is AWS's documentation account, `111122223333`, and every id is made up. Each prompt or tool text
carries the canary string `F036-CANARY-PROMPT`, so a test can show that no text reaches a stored
record. They are to be replaced by redacted live captures once the integration account's A/B test exists.
