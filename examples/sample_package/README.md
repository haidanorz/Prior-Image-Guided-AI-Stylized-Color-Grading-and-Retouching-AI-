# Feature fingerprint package

This directory is the offline encoder output. It contains no XMP, no source images,
and no per-image analysis text.

- `fingerprint.json`: runtime fingerprint (bucket parameters; global as fallback)
- `global_signature.json`, `profile.json`, `buckets/*.json`: evidence for audit
- `package_manifest.json`: per-file SHA-256 and package hash

Status: draft / application_ready=false
Fingerprint schema: 1.1.0
Fingerprint id: fp_2c4d86fbbcd22f19
Profile shape: bucket_primary

Runtime rule: classify the target scene first, then use the bucket parameters;
`global` is a fallback for keys the bucket does not carry, or for unknown scenes.
Never blend two buckets.

Integrity:
1. Recompute SHA-256 for each file in `package_manifest.json`.
2. Recompute package_hash from `files` + `fingerprint_hash`.
3. Recompute `fingerprint_hash` from `fingerprint.json` with the hash field removed.
