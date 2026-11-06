[简体中文](README.md) | **English**

# AI-Driven Stylized Color Grading from Reference Images

## What this project does

After a shoot, photos usually still need to be graded one by one in Photoshop or Lightroom before they look right — a slow, repetitive job.

Yet a large share of those photos repeat the same scenes and the same grading ideas: cosplay sets from conventions, travel landscapes from different trips, and so on.

This project provides an AI-based approach: it extracts a reusable grading style from your previous body of work and applies it to new photos automatically, removing most of the repetitive grading work.

Technically, it extracts an auditable grading prior from a set of **already-graded reference photos**, then drives a **new photo** through several rounds of real Lightroom rendering feedback until it converges on that photo's own best Develop settings.

It fits the case where you have a body of work you like and want to carry that handling habit onto new photos; it also fits engineering settings where grading decisions must leave an evidence trail.

---

## Why use a language / multimodal model for a pure vision task

There is no shortage of automated grading projects built on CV models and deep learning on GitHub and in journals. This project takes another path, based on the author's view:

The skill of photo grading is passed down through mentorship and learned by watching and discussing — its carrier is verbal and visual knowledge, not large-scale parameter statistics or mathematical fitting. Grading is itself a benchmark based on subjective aesthetics, with no consistent standard. The author therefore argues that this kind of verbally grounded skill acquisition is better served by a large model's language ability and memory of knowledge than by CV algorithms.

A second advantage of building on multimodal large models: the model interface can be swapped for any model API (ChatGPT, DeepSeek, and others). As model capability improves, the results of this pipeline may improve with it. Traditional CV architectures are hard to upgrade and can only be fine-tuned, whereas here the core capability module can be replaced outright.

---

## Quick start

Shortest path; only Python is required. Lightroom is needed only when you actually export a photo.

```bash
# 1) Clone and install (Python >= 3.10)
git clone <repo-url>
cd <repo-dir>
python -m pip install -r requirements.txt

# 2) Verify the bundled sample fingerprint package
python -m online_decode verify-package --package examples/sample_package

# 3) Run the tests to confirm the environment (no Lightroom needed)
python -m unittest discover -s online_decode/tests -t .
```

The first three steps take about a minute. To run a real photo, point `--photo` at your own file:

```bash
python -m online_decode run \
  --photo "D:/photos/IMG_0001.jpg" \
  --package examples/sample_package \
  --workspace "D:/decode_sessions" \
  --max-rounds 5 \
  --intent "keep sky separation"
```

The session prints the path of `agent_io/req_*.json`; write a matching `resp_*.json` to move on.
After an interruption, `python -m online_decode resume --session <session_id>` continues without
redoing finished rounds.

To use your own style set, run `offline_preprocess` → `offline_profile` →
`offline_profile package` in order. Full commands and flags are in
[Section 3](#3-getting-started); the Lightroom plugin required for real exports is in
[Section 4](#4-lightroom-feedback-dependency).

---

## 1. Three stages

```text
reference photos ──► offline_preprocess ──► previews + objective features + manifest
                                                    │
                                                    ▼
                                        offline_profile (encode)
                            per-image analysis → scene buckets → robust statistics
                                                    │
                                                    ▼
                                    fingerprint package (draft, hash-verifiable)
                                                    │
new photo ──────────────────────────────────────────┤
                                                    ▼
                                        online_decode (decode)
                   scene → brief → n rounds of plan/evaluate
                          → pick best round → restore → export
```

| Stage | Package | Input | Output |
|---|---|---|---|
| Preprocessing | `offline_preprocess` | Folder of finished photos | sRGB previews, objective feature JSON, manifest/errors |
| Encoding (offline profiling) | `offline_profile` | Preprocess output + per-image analysis responses | Global signature, scene bucket parameters, bucket narratives, fingerprint package |
| Decoding (online grading) | `online_decode` | New photo + fingerprint package + Lightroom feedback port | Per-round XMP, evaluations, best-round snapshot, final image and full report |

### Encoding: where the priors come from

1. An agent looks at each reference photo and produces structured scene observation, visual observation and edit hypotheses; responses are normalized before being stored.
2. Photos are bucketed by scene (`natural_landscape` / `urban_architecture` / `street` / `indoor` / `still_life`): core samples enter a bucket, marginal ones are only associated.
3. Within a bucket, robust statistics are computed: median, interquartile range, support, sign agreement. Items stable across buckets are promoted to the global signature; bucket-specific items become conditional deltas; items that disperse too widely are marked `prior_only` and never become hard constraints.
4. A bucket narrative is induced from that bucket's description set, so decoding can understand "the language of this bucket" instead of applying parameters mechanically.
5. Export the fingerprint package: per-file SHA-256 plus `package_hash`; any modified file fails verification.

Package layout:

```text
sample_package/
├── package_manifest.json      # per-file hashes + package_hash
├── fingerprint.json           # the only file the decoder needs at runtime
├── global_signature.json      # global signature
├── profile.json               # build metadata
├── README.md
└── buckets/<bucket_id>/
    ├── profile.json           # bucket parameters (published / wide_dispersion / withheld)
    ├── delta.json             # delta against the global signature
    ├── stats.json             # quantiles and dispersion
    └── narrative.json         # bucket narrative
```

### Decoding: how the loop runs

1. `scene`: the original image only, to classify the scene bucket; bucket parameters stay hidden in this round so the prior cannot contaminate classification.
2. `brief`: the bucket anchors, wide-dispersion priors and narrative are provided; the model returns `preserve / goals / avoid / initial_xmp`.
3. `plan`: from round 2 on, a full XMP; the controller enforces the field whitelist, value ranges, reference band and type checks, and records every rejection in the ledger.
4. `evaluate`: each round is rendered by real Lightroom; the model gives a verdict and improvement score against verifiable metrics; the loop stops once improvement falls below the threshold.
5. Select the best round (not the last one), restore its snapshot and export at full size; a round only counts when the hash chain is closed.

---

## 2. Layout

```text
.
├── offline_preprocess/     # preprocessing: color management, resizing, features, pHash, verification
├── offline_profile/        # offline profiling: analysis protocol, bucketing, statistics, narratives, packages
├── online_decode/          # online decoding: state machine, parameter guards, XMP, LR feedback, MCP tools
├── examples/
│   ├── make_sample_package.py   # generate the synthetic sample package
│   └── sample_package/          # generated sample package (fully synthetic data)
├── requirements.txt
├── requirements-mcp.txt
└── LICENSE
```

Each module has its own `README.md` with detailed notes.

---

## 3. Getting started

```bash
python -m pip install -r requirements.txt      # core deps: numpy / Pillow / jsonschema
```

### 3.1 Verify the sample fingerprint package

```bash
python -m online_decode verify-package --package examples/sample_package
```

### 3.2 Run the tests (no Lightroom needed)

```bash
python -m unittest discover -s offline_preprocess/tests -t .
python -m unittest discover -s offline_profile/tests -t .
python -m unittest discover -s online_decode/tests -t .
```

230 tests in total. `online_decode` uses a fake LR port and scripted drivers to cover the state
machine, guards, hash chain, early stopping, resume and Golden replay.

### 3.3 Encode your own reference set

```bash
# 1) Preprocess: previews + objective features + manifest
python -m offline_preprocess \
  --input-dir /path/to/finished_photos \
  --output-dir ./build/preprocess

# 2) Per-image analysis → bucketing → statistics → build
#    --mock-dir points at frozen analysis responses (format: the analysis contract in offline_profile/README.md)
python -m offline_profile \
  --input ./build/preprocess \
  --build-root ./build/profile \
  --mock-dir ./responses

# 3) Verify and export the fingerprint package
python -m offline_profile verify --build-root ./build/profile
python -m offline_profile package --build-root ./build/profile --out-dir ./build/package
```

Without `--mock-dir`, the task-orchestration tools in `offline_profile/mcp_server.py`
(`profile_start` → `profile_tasks` → `profile_submit` → `profile_finalize`) let an agent submit
one analysis per image, after which you run `verify` / `package` as above.

### 3.4 Decode a new photo

```bash
python -m online_decode run \
  --photo "D:/photos/IMG_0001.jpg" \
  --package examples/sample_package \
  --workspace "D:/decode_sessions" \
  --max-rounds 5 \
  --intent "keep sky separation"
```

Outputs land in `workspace/sessions/<session_id>/`: `session.json` (single source of truth),
`rounds/rNN/` (model XMP, validated XMP, ledger, snapshot, metrics, evaluation, preview) and
`final/` (final image and best snapshot).

### 3.5 Use it from an MCP host

```bash
python -m pip install -r requirements-mcp.txt
```

Register the two servers in your MCP client (use real absolute paths):

```json
{
  "mcpServers": {
    "offline-profile": { "command": "python", "args": ["/ABS/PATH/TO/REPO/offline_profile/mcp_server.py"] },
    "online-decode":  { "command": "python", "args": ["/ABS/PATH/TO/REPO/online_decode/mcp_server.py"] }
  }
}
```

Environment variables for online decoding:

| Variable | Purpose |
|---|---|
| `ONLINE_DECODE_PACKAGE` | Default fingerprint package (falls back to `examples/sample_package`) |
| `ONLINE_DECODE_MCP_REPO` | Plugin MCP directory (required for real Lightroom feedback) |
| `ONLINE_DECODE_WORKSPACE` | Session directory (default `~/Pictures/Lightroom/online_decode_sessions`) |
| `ONLINE_DECODE_VISION_KEY` | Optional; enables the OpenAI-compatible vision API path for `decode_run` |

---

## 4. Lightroom feedback dependency

The `evaluate` stage of online decoding needs real exports and readings from Lightroom, provided
by a companion plugin:

- Repository: [AI-generated-xmp-file-interface-plugin-for-lightroom-cc](https://github.com/haidanorz/AI-generated-xmp-file-interface-plugin-for-lightroom-cc)
- Components: Lightroom Classic plugin (apply Develop settings / render previews / restore snapshots / export) plus a local MCP interface
- Without the plugin you can still run every unit test and Golden replay (fake port), but no real image can be produced

---

## 5. Scope and honest claims

- Fingerprint packages default to `status=draft`, `application_ready=false`; online decoding should treat them as a **pilot prior** and makes no claim of reproducing a photographer's stable style.
- Decoding is not guaranteed to be reproducible: two runs on the same photo may take different rounds. That is expected behavior of a feedback loop, not a defect.
- Parameter writes are gated: fields outside the whitelist, type mismatches, values beyond the reference band, and fields the interface cannot write are all rejected, with the reason recorded item by item.
- Tone curves and local masks are outside the model's control; use snapshot restore when needed.
- All image processing only reads inputs and writes to output directories; previews strip EXIF/GPS by default.

## 6. License

MIT, see [LICENSE](LICENSE).
