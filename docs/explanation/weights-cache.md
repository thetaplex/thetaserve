# The weights cache

A 27-31 B model has 29-31 GiB of weights at 8 bits. When vLLM starts, it downloads them from
Hugging Face, unless they are already on disk. This page explains where that disk is, and what
it costs to keep it between runs.

## What Runpod offers

Prices and behaviour as Runpod documents them on 2026-10-20.

| Storage | Kept after `stop` | Kept after `down` | Tied to | Price |
| --- | --- | --- | --- | --- |
| Container disk | no | no | the pod | $0.10/GB/month |
| Pod volume disk | yes | no | the pod | $0.10/GB/month running, $0.20 stopped |
| Network volume | yes | yes | one data centre | $0.07/GB/month |
| Global volume (beta) | yes | yes | nothing | $0.09/GB/month plus IOPS |

## Why thetaserve uses a network volume

By default, thetaserve keeps the weights on a network volume, `thetaserve-cache`. Thus a second
launch does not download them again. This has a cost in addition to the storage bill: a network
volume is in one data centre, and each pod that uses it must run there. When the GPU is sold
out in that data centre, the cache is not usable, and `up` fails with "sold out where your
volume lives".

## Why the volume is optional

Two numbers decide if the cache is worth its cost: the time of a cold download, and how often
you launch. A download of a few minutes on an A40 at $0.59/h costs a few cents of GPU time.
100 GB of network volume costs $7 a month, if you launch or not. If you launch a few times a
month, you pay less without the volume, and you can use any data centre.

`network_volume: null` removes the volume, its mount and the data-centre pin. But then
`stop` and `start` save nothing, because `stop` erases the container disk. End each run with
`down`.

## Not yet measured

- **The cold-start time** without a cache.
- **Global volumes** remove the data-centre pin and keep a cache. They are in beta, and the
  IOPS cost of a 29 GiB read for each launch is not known. thetaserve does not use them.
