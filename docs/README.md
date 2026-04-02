WIP 02/04/26

To-do:
* points 3/4
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
2. Activate the new environment (if you kept its default name) with `conda activate wakee_rldd`.
3. `cd` to the path where you've extracted the project; it should end with `WAKEE.reloaded`.
4. Make sure Docker is running and ready to build.
5. Finally, create and fill your `.env` file ( :heavy_exclamation_mark: never share your credentials!) by following the example given in `.env_example.md` as follows:

* [AWS](https://aws.amazon.com/fr/): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (your user credentials to access AWS) --- S3_BUCKET_URI (the full s3 URI of your bucket: `s3://...`) --- S3_BUCKET_NAME, S3_WORK_FOLDER (from your S3 interface, exclusively the *names* of your bucket and the working folder contained within: `my-wakee-bucket` & `my-working-folder`) --- ARTIFACT_STORE_URI (the full s3 URI under which ML artifacts will be stored: `s3://...`)
* [NeonDB](https://neon.com/docs/get-started/connect-neon) (or any host for your relational database, we used PostgreSQL): SQL_NEONDB_URI (optional: **deprecated** back-up URI `postgresql://...`) --- BACKEND_STORE_URI (full URI necessary to run MLflow: `postgresql://`)
* [Mistral API](https://docs.mistral.ai/getting-started/quickstart): MISTRAL_API_KEY (your private API key)
* [Evidently AI](https://docs.evidentlyai.com/quickstart_ml): EVIDENTLY_API_KEY (your private API key) --- EVIDENTLY_AI_PROJECT_ID (the unique ID of the Evidently project enabling monitoring)
* MLflow (we hosted ours on a [HuggingFace space](https://huggingface.co/docs/hub/spaces)): TRACKING_SERVER_URI (full address: `https://...`)

---

### 2. Understanding the architecture [:top:](#wakeereloaded-documentation)

![](architecture.png)

WAKEE.reloaded incorporates multiple elements, some being available only locally, others in the cloud.

WAKEE in itself is meant to run locally for multiple reasons; chief amongst them being, to prevent the ease in porting it to a convenient live application, going against regulations.

When we refer only to "WAKEE", it is the script which can only be run on your localhost, bringing your web browser up. This is where the user begins: the convenient interface enables them to start or stop the real-time capture through their webcam, setting their work sessions on the left-hand side, and getting recommendations (upon detection of cognitive drift) on the right-hand side.

By deep learning means (through a convolutional neural network), emotions are recognized; when appropriate, LLM prompt engineering produces recommendations to help the user focus again.

Whenever the user turns off their webcam, WAKEE will prompt them with one of the captured frames, asking them to rate their own emotions. This is where we get our new labeled data through human feedback!

Then begins the part that the user is not expected to be aware of: WAKEE.reloaded, involving the variety of functionalities making the model live behind our project.

Airflow orchestration begins with a daily DAG, enabling the writing of a .csv file associating the labels provided by the users to their .jpg frame. Should the frame miss its ratings, it will be deleted; otherwise it is hosted on a S3 bucket as-is.

This ends the daily orchestration (lower left on the schema); it is only meant to prepare the weekly training, while removing daily any leftover and/or unused data.

Then begins the first half of the weekly DAG (upper left on schema). The new data gathered daily is compiled once a week, before starting a new model training; its results are stored into an MLflow run, allowing us to track its results.

Should the new challenger model perform better than the champion in use, this is where the second half of the weekly DAG (center of the schema) starts: the new model is uploaded in the dedicated repository of WAKEE.reloaded. Using GitHub Actions, test scripts are run; should they all be passed, the continuous deployment part of the project will go through by updating all models in use.

As they deteriorate over time, Evidently AI will be put to practice to monitor how WAKEE's model performs, and will raise the alert in case of something crossing thresholds.

Now completely out of the DAGs, the end user is able to make use of the latest update of WAKEE's model!

Following our certificate's instructions, we were also expected to deploy an API online to serve our model and provide documentation for it. Though technically available to meet this criteria on a [HuggingFace space](https://mevelios-wakee-reloaded-api.hf.space/), it remains unadvertised on the root README for obvious reasons. Running on free resources, the space may take up to 120 seconds to restart; it won't permit real-time detection as it tolerates at most two calls per minute. Data sent to the API endpoint will not be stored; it is only meant to be run on the model to return a result, then discarded.

---

### 3. Starting the system [:top:](#wakeereloaded-documentation)

---

### 4. Limits and improvements [:top:](#wakeereloaded-documentation)

data poisoning (from feedback)