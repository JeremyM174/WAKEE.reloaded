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

Given the topic of regulations, we go by the principle the reader is aware of their basic use to (hopefully) prevent the misuse by non-initiated users and prevent any headache consequently.

1. To build your environment (we used Anaconda Navigator to manage it), you may find a convenient `wakee_rldd_backup.yaml` file.
2. Make sure Docker is running and ready to build.
3. Activate the new environment (if you kept its default name) with `conda activate wakee_rldd`.
4. `cd` to the path where you've extracted the project; it should end with `WAKEE.reloaded`.

Please note WAKEE.reloaded is heavily impacted by performance: given the real-time detection, it is meant to run on computers with higher specs - at least having a GPU. Obviously, it also requires having a webcam and enabling access to it through its local application (remember to check permissions within both your operating system and your internet browser).

---

### 2. Understanding the architecture [:top:](#wakeereloaded-documentation)

![](architecture.png)

---

### 3. Starting the system [:top:](#wakeereloaded-documentation)

---

### 4. Limits and improvements [:top:](#wakeereloaded-documentation)

.