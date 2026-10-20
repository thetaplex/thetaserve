# The weights cache

A 27-31 B model is 29-31 GiB of weights at 8 bits. vLLM downloads them from Hugging Face when it
starts, unless they are already on disk. The question is where "on disk" lives, and what it
costs to keep it there between runs.

## What Runpod offers

Prices and behaviour as documented by Runpod on 2026-10-20.

| Storage | Survives `stop` | Survives `down` | Tied to | Price |
| --- | --- | --- | --- | --- |
| Container disk | no | no | the pod | $0.10/GB/month |
| Pod volume disk | yes | no | the pod | $0.10/GB/month running, $0.20 stopped |
| Network volume | yes | yes | one data centre | $0.07/GB/month |
| Global volume (beta) | yes | yes | nothing | $0.09/GB/month plus IOPS |

## Why thetaserve used a network volume

M1 caches weights on a network volume, `thetaserve-cache`, so a second launch skips the download.
That choice has a cost beyond the storage bill: a network volume lives in one data centre, so
every pod that uses it must run there. When the GPU sells out in that data centre, the cache is
useless and `up` fails with "sold out where your volume lives". The automatic GPU choice in
[ADR 0001](../adr/0001-derive-launch-options.md) has to rank data centres by "already holds the
cache" for the same reason.

## Why it is now optional

Whether the cache pays for itself depends on two numbers: how long a cold download takes, and
how often you launch. A download that takes a few minutes on an A40 at $0.59/h costs a few cents
of GPU time. 100 GB of network volume costs $7 a month whether or not you launch. Someone who
launches a few times a month is better off without it, and gains free choice of data centre.

Setting `network_volume: null` removes the volume, its mount and the data-centre pin. The price
is that `stop`/`start` no longer saves anything, because the container disk is wiped on stop; end
runs with `down`.

## Still open

- **The cold-start time** is unmeasured. The M1 pilot runs without a cache to measure it.
- **Global volumes** remove the data-centre pin while keeping a cache. They are in beta and
  their IOPS charge for a 29 GiB read per launch is unknown.

The default stays a network volume until those are measured.
