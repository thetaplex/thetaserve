# thetaserve documentation

Organised by [Diátaxis](https://diataxis.fr/): each page is one kind of document. Dates are
ISO 8601; memory sizes are GiB (IEC 80000-13) unless a provider quotes GB.

| Kind | Page | Read it when |
| --- | --- | --- |
| Tutorial | [Your first deployment on Runpod](tutorials/first-deployment.md) | You have never run thetaserve |
| How-to | [Store credentials](how-to/store-credentials.md) | Setting up a machine or a Runpod API key |
| How-to | [Override the catalogue](how-to/override-catalogue.md) | A GPU or model you need is missing, or a value is `TODO` |
| How-to | [Run without a cache volume](how-to/run-without-cache-volume.md) | You do not want storage billed between runs |
| How-to | [Connect a client or coding agent](how-to/connect-a-client.md) | A deployment is ready |
| Reference | [CLI](reference/cli.md) | Looking up a command or option |
| Reference | [Configuration and files](reference/configuration.md) | Looking up a catalogue field, a path or the secrets lookup order |
| Explanation | [The weights cache](explanation/weights-cache.md) | Deciding whether to keep a cache volume |
| Explanation | [How launch settings are chosen](explanation/launch-settings.md) | Wondering why tiers exist and where they are going |

Design records: [DESIGN.md](DESIGN.md) (the design brief) and [ADRs](adr/).
