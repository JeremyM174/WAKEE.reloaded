WIP 02/04/26

To-do:
* points 2/3/4
* technical stack presentation +afterword?

# <p align="center">WAKEE.reloaded documentation</p>

### Table of contents

1. [Pre-requisites](#1-pre-requisites-top)
2. [Understanding the architecture](#2-understanding-the-architecture-top)
3. [Starting the system](#3-starting-the-system-top)
4. [Limits and improvements](#4-limits-and-improvements-top)

> [!WARNING]
> As a reminder by European standards of the AI Act - [article 5.1.f](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_5), putting either WAKEE, either WAKEE.reloaded into production is **strictly forbidden** and becomes **your** sole responsibility. Doing so **isn't intended** by any of us; sharing our project is only meant to help us with work applications, as well as a requirement for our French certificate RNCP38777 "AI Architect"!

![](WAKEE_image.png)

---

### 1. Pre-requisites [:top:](#wakeereloaded-documentation)

WAKEE.reloaded is meant to run with:
* Anaconda Navigator,
* Anaconda Prompt,
* Docker.

Given the topic of regulations, we go by the principle the reader is aware of their purpose and use to (hopefully) prevent the misuse by at least some non-initiated users, to prevent any headache consequently.

Please note WAKEE.reloaded is heavily impacted by performance: given the real-time detection, it is meant to run on computers with higher specs - at least having a GPU. Obviously, it also requires having a webcam and enabling access to it through its local application (remember to check permissions within both your operating system and your internet browser).

Now for the actual steps to be performed before any use:

1. To build your environment (we used Anaconda Navigator to manage it), you may find a convenient `wakee_rldd_backup.yaml` file.
2. Make sure Docker is running and ready to build.
3. Activate the new environment (if you kept its default name) with `conda activate wakee_rldd`.
4. `cd` to the path where you've extracted the project; it should end with `WAKEE.reloaded`.
5. Finally, create and fill your `.env` file ( :heavy_exclamation_mark: never share your credentials!) by following the example given in `.env_example.md` as follows:

* [AWS](https://aws.amazon.com/fr/): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (your user credentials to access AWS) --- S3_BUCKET_URI (the full s3 URI of your bucket: `s3://...`) --- S3_BUCKET_NAME, S3_WORK_FOLDER (from your S3 interface, exclusively the *names* of your bucket and the working folder contained within: `my-wakee-bucket` & `my-working-folder`) --- ARTIFACT_STORE_URI (the full s3 URI under which ML artifacts will be stored: `s3://...`)
* [NeonDB](https://neon.com/docs/get-started/connect-neon) (or any host for your relational database, we used PostgreSQL): SQL_NEONDB_URI (optional: **deprecated** back-up URI `postgresql://...`) --- BACKEND_STORE_URI (full URI necessary to run MLflow: `postgresql://`)
* [Mistral API](https://docs.mistral.ai/getting-started/quickstart): MISTRAL_API_KEY
* [Evidently AI](https://docs.evidentlyai.com/quickstart_ml): EVIDENTLY_API_KEY, EVIDENTLY_AI_PROJECT_ID
* MLflow (we hosted ours on a [HuggingFace space](https://huggingface.co/docs/hub/spaces)): TRACKING_SERVER_URI (full address: `https://mevelios-mlflowtest.hf.space`)

---

### 2. Understanding the architecture [:top:](#wakeereloaded-documentation)

![](architecture.png)

---

### 3. Starting the system [:top:](#wakeereloaded-documentation)

---

### 4. Limits and improvements [:top:](#wakeereloaded-documentation)

.