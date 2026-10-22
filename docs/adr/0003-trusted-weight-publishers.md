# 3. Trust weight publishers explicitly and pin the commit

Date: 2026-10-22

## Status

Proposed. Point 3 (pinning) is implemented; points 1 and 2 wait for acceptance. Amends
[ADR 0001](0001-derive-launch-options.md), point 3 (model candidates).

## Context

ADR 0001 takes weight variants from the Hugging Face Hub filter `base_model:quantized:<repo>`,
most downloaded first. Anyone can publish a repository with that tag, and download counts are
easy to inflate, so the variant `up` suggests can come from any account.

What a variant can and cannot do, checked on 2026-10-22:

- `.safetensors` files hold only tensors; loading them runs no code. Model-supplied Python runs
  only with `--trust-remote-code`, which thetaserve never passes.
- The weights themselves can be changed before quantising: a backdoor or removed safety
  training is not visible in any metadata.
- `chat_template.jinja` and `tokenizer_config.json` shape every prompt and tool call. vLLM
  renders them sandboxed, but a changed template can still change behaviour, for example by
  adding a system prompt.
- vLLM downloads `main` at start-up, so a push after the user chose an option changes what runs.

An example from the Gemma 4 26B list: `farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit` (about
123,000 downloads) is a fine-tune tagged as a quantisation. It ranks below
`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` today and is not suggested, but nothing prevents it.

Reputable quantised variants exist for the launch models: from the model's own organisation
(`google`, `Qwen`), from `RedHatAI` (Red Hat's verified organisation, maintainers of vLLM's
`llm-compressor`) and from `nvidia`. Some 4-bit AWQ variants exist only from individuals.

Alternatives considered:

- **Keep any publisher, rank by downloads.** This gives the most options, but the ranking is
  something an attacker controls.
- **Show the publisher and let the user judge.** This is honest but does not help with `--yes`
  or with users who do not know the organisations.
- **Verify weights against the base model**, for example by comparing outputs. This is the only
  check on the weights themselves, but it needs a GPU and a test set, which makes it a `bench`
  feature, not a launch check.

## Decision

1. **Trusted publishers.** `up` considers only variants whose organisation is the base model's
   own or is listed in a packaged `publishers.yaml`, initially `RedHatAI` and `nvidia`. Users
   add organisations in their config directory (DESIGN.md section 3).
2. **Opt-in for the rest.** `--any-publisher` lists other publishers' variants as well, marked
   in the table.
3. **Pinned commit.** When `up` reads a variant's file sizes, it records the commit hash of that
   revision and passes `--revision <hash>` to `vllm serve`. vLLM defaults the tokenizer revision
   to the model revision, so the tokenizer and chat template are pinned too.

## Consequences

- Fewer 4-bit options by default: for Gemma 4 26B the AWQ rows disappear unless
  `--any-publisher` is passed or the publisher is added.
- The weights, tokenizer and chat template that run are the ones whose sizes were checked, and
  a launch command can be repeated exactly.
- A trusted publisher can still publish bad weights; this limits whom the user trusts, it does
  not verify anything. Checking weights stays a possible `bench` feature.
- Launches pinned with `--device` use the catalogue's repository at `main`, unpinned, until
  that path is removed (ADR 0001).
