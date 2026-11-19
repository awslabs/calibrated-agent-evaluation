# AgentCore dataset API responses

Read-only responses from an AgentCore account (us-west-2), captured on 2026-10-27 with
botocore 1.43.106, for the offline dataset publish and pull tests (botocore `Stubber`).

- `list_datasets.json`, `get_dataset_third_party.json`, `list_dataset_versions_none.json`,
  `list_dataset_examples_third_party.json`: a `THIRD_PARTY_EVALUATION_V1` dataset the CloudWatch
  console made, the only one the account held. Listed examples carry the service-added `exampleId`.
- `get_dataset_missing_version_error.json`: `GetDataset` for a version that does not exist
  (`ResourceNotFoundException`, HTTP 404).

Redactions: the account id is `111122223333`, the dataset id is `test-AbCdEf1234`, the space-id tag
and the example and scenario ids are synthetic UUIDs, the presigned `downloadUrl` is
`https://example.invalid/download`, and every example text value is replaced by `<field name>`.
`ResponseMetadata` is dropped. The write operations have no captured responses yet: their stubs
are built from the botocore output shapes (botocore's `Stubber` validates each one), and the
opt-in live test exercises the real calls.
