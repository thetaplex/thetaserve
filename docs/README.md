# thetaserve documentation

Organised by [Diátaxis](https://diataxis.fr/): each page is one kind of document. Dates are
ISO 8601; memory sizes are GiB (IEC 80000-13) unless a provider quotes GB.

| Kind | Page | Read it when |
| --- | --- | --- |
| Tutorial | [Your first deployment on Runpod](tutorials/first-deployment.md) | You have never run thetaserve |
| How-to | [Store credentials](how-to/store-credentials.md) | Setting up a machine or a Runpod API key |
| How-to | [Override the catalogue](how-to/override-catalogue.md) | You need a new preset, a pinned vLLM image, or a new model name |
| How-to | [Run without a cache volume](how-to/run-without-cache-volume.md) | You do not want storage billed between runs |
| How-to | [Connect a client or coding agent](how-to/connect-a-client.md) | A deployment is ready |
| How-to | [Launch on Runpod serverless](how-to/launch-on-serverless.md) | You want to pay only while requests are served |
| Reference | [CLI](reference/cli.md) | Looking up a command or option |
| Reference | [Configuration and files](reference/configuration.md) | Looking up a catalogue field, a path or the secrets lookup order |
| Reference | [M4 benchmark plan](reference/benchmark-plan.md) | Looking up which runs M4 makes and what each measures |
| Explanation | [The weights cache](explanation/weights-cache.md) | Deciding whether to keep a cache volume |
| Explanation | [How launch settings are chosen](explanation/launch-settings.md) | You want to know why `up` shows these options |

Design: [DESIGN.md](DESIGN.md) is the design brief. Its section 17 is the decision log.

Writing rules: text follows [ASD-STE100](https://www.asd-ste100.org/) Simplified Technical
English where it applies. Procedures use the imperative, one instruction per sentence.
