# Lab Submission Instructions

## Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:**

**Member 1:**

| **Details**                                                                                                                           | **Comment**       |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------------|
| **Student ID**                                                                                                                        | **161749**        |
| **Name**                                                                                                                              | **Maddox Kibati** |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Clustering endpoint, Docker, Nginx, Render, video, and integration. Built the farmer-segmenter endpoint and the k-Means artifact (PR #1: https://github.com/BI-course/smm-202604-D1/pull/1), ran the API in Docker with Gunicorn behind an Nginx reverse proxy over HTTPS, deployed the unmodified `api.py` to Render, and recorded the demonstration video. When the Docker image crashed on start-up, I read the container logs, traced it to model files saved with an older scikit-learn and with pandas 3 Arrow-backed strings, fixed the notebook, and re-saved the regression, classification and recommender model files under the course library versions (PR #4). I also completed this submission file. Branches: https://github.com/BI-course/smm-202604-D1/tree/feature/lab-smm/clustering-endpoint , https://github.com/BI-course/smm-202604-D1/tree/feature/lab-smm/resave-models and https://github.com/BI-course/smm-202604-D1/tree/feature/lab-smm/docker-nginx-render . Learned: models must be saved with the same library versions that serve them, or the container crashes at start-up, and that Gunicorn behind Nginx is what lets the same code serve real traffic instead of the Flask development server.**              |

**Member 2:**

| **Details**                                                                                                                           | **Comment**        |
|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------------|
| **Student ID**                                                                                                                        | **155578**         |
| **Name**                                                                                                                              | **Justus Kamande** |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Regression and classification endpoints. Added the farm-income-regressor and invoice-risk-classifier endpoints to `api.py` with input validation, replacing the SME sample models with the group's models (PR #3: https://github.com/BI-course/smm-202604-D1/pull/3). Both endpoints return correct predictions locally, in Docker behind Nginx, and on Render. The saved model files were later re-saved by Maddox under the course library versions after they failed to load in the Docker image (branch resave-models, PR #4). Branch: https://github.com/BI-course/smm-202604-D1/tree/feature/lab-smm/regression-classification . Learned: a saved pipeline carries its own imputer, encoder and scaler, so the endpoint only validates the JSON and calls predict, and because the target was log-transformed the output must be converted back with expm1.**               |

**Member 3:**

| **Details**                                                                                                                           | **Comment**         |
|:--------------------------------------------------------------------------------------------------------------------------------------|:--------------------|
| **Student ID**                                                                                                                        | **151115**          |
| **Name**                                                                                                                              | **Tyrone Odhiambo** |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | **Recommender endpoint. Added the menu-recommender and menu-items endpoints to `api.py`, replacing the sample ecommerce recommender with the group's association rules lookup (PR #2: https://github.com/BI-course/smm-202604-D1/pull/2). The endpoint accepts item ids or names and returns a clear 400 error for an empty cart or an unknown item. The rules file was later re-saved by Maddox without Arrow-backed strings after it failed to load in the Docker image (branch resave-models, PR #4). Branch: https://github.com/BI-course/smm-202604-D1/tree/feature/lab-smm/recommender-endpoint . Learned: a recommender needs no predict call because the mined rules become a lookup table, and a recommendation only appears when the cart contains a full rule antecedent**                |


## Chosen Level of Difficulty

**Specify the chosen level of difficulty** (baseline, intermediate, or advanced): **Intermediate**

## Video Demonstration

Submit the link to a short video (**not more than 10 minutes**) demonstrating your complete solution at whichever difficulty level you chose. Please ensure that the lecturer has rights to view the video.

At minimum, the video must show, in one continuous walkthrough:

1. `api.py` running **locally**, with a live request/response against each model endpoint you implemented.
2. If you attempted the Intermediate or Advanced tier: the same API running inside its **Gunicorn + Nginx Docker container**, reached through the reverse proxy, not the Flask development server.
3. If you attempted the Advanced tier: your **Hugging Face Space** and your **Streamlit Community Cloud app**, each opened live in a browser, with at least one prediction made on each tab, and your **Render**-hosted API answering a request from Postman or `curl`.

**Why this matters for grading:** free-tier public deployments are not guaranteed to still exist by the time this is graded. The video is treated as the authoritative record of what was actually working at submission time; a broken link at grading time is not penalized if the video clearly shows that deployment working.

Note that you are required to submit the link to the video and NOT the video itself. The video should NOT be uploaded to your repository—that would be a misuse of GitHub.

**Link to the video:** **https://drive.google.com/file/d/10ehY-icFie3dpVTgCk7u4BzIB1l8-nCm/view?usp=sharing**

## Public URLs

Fill in only the rows that apply to your chosen difficulty level (see Part C below). Leave a row blank if your tier does not require that deployment target.

| **Deployment target**                             | **URL** |
|:--------------------------------------------------|:--------|
| Render (the full `api.py`, all endpoints)         | **https://smm-202604-d1.onrender.com**    |
| Hugging Face Space (Gradio, one tab per model)    |         |
| Streamlit Community Cloud app (one tab per model) |         |

---

## Scenario

Across earlier labs, **your group** was assigned its own dataset and trained your own models for four distinct tasks:

1. A **regression** task
2. A **classification** task
3. A **k-Means clustering** task
4. An **Apriori association-rule mining** task

Different groups worked with different datasets and, in some cases, different algorithms for each task -- there is no single shared reference repository or model list for the whole class. Wherever this document refers to "your regressor," "your classifier," "your clustering model," or "your recommender," it means **whichever model your own group already trained and saved for that task**, not a specific named algorithm.

You are now required to build a single Flask API (`api.py`) that serves **your own group's four models**, loaded from disk (never retrained inside `api.py`), and to demonstrate that API running correctly across a growing number of deployment surfaces depending on the difficulty tier you choose.

### Part A: Regression and Classification

- Locate the trained regression and classification model files your group produced and saved in your earlier labs on Regression and Classification.
- Update [api.py](api.py) to include end-points that load these two models from disk and serve predictions from them.

### Part B: Clustering and Association Rule Mining

- Locate the trained clustering model and the association rules your group produced and saved in your earlier labs on Clustering and Association Rule Mining.
- Update [api.py](api.py) to include end-points that:
  - Load the clustering model from disk and predict which cluster a new client belongs to
  - Load the association rules from disk and recommend products based on them

**Note 1:** **`api.py` is NOT production-grade as it is.** It is only meant for demonstration purposes. Scalability and security must be taken into consideration before deploying an API in a production environment.

**Note 2:** Some students often treat the API as an afterthought, focusing only on training ML models. In practice, **the API is the product** -- it is how others interact with your model. The "hidden" learning here is that the delivery mechanism (API design, usability, error handling, and even documentation) often matters more to stakeholders in the industry than the models themselves.

### Part C: Deployment Breadth (Read Before Choosing Your Tier)

All three tiers below ask you to serve **the same four models** (your regressor, your classifier, your clustering model, your recommender) from **one, unmodified `api.py`**. The model files and the Flask routes do not change between tiers -- only how many different places that one working API is shown to run correctly.

**Baseline (Required):**

- Update `api.py` to serve **at least three of your four** models, loaded from disk.
- Run and demonstrate `api.py` **locally** with Flask's development server, with a successful request against each implemented endpoint.

**Intermediate (Recommended):**

- Update `api.py` to serve **all four** of your models, loaded from disk.
- Dockerize your Flask API using **Gunicorn**, and place it behind an **Nginx** reverse proxy (see the reverse-proxy lab notes for the full Docker Compose setup).
- Deploy the same, unmodified `api.py` to **Render**, and demonstrate a real request against it from outside your own machine (cURL or Postman).

**Advanced (Optional):**

- Everything required for Intermediate, plus:
  - Create a web page (or pages) using Basic HTML, CSS, and Vanilla JavaScript that demonstrates the use of the API
  - Implement basic error handling (e.g., missing inputs)
  - Build **one Gradio app with a tab per model** (all four of your models) and publish it as a **Hugging Face Space**
  - Build **one Streamlit app with a tab per model**, covering the same four models, and publish it on **Streamlit Community Cloud**
  - All deployment targets -- local, Dockerized/Nginx, Render, Hugging Face, and Streamlit -- must serve models loaded **from the same trained artifacts**; retraining different models per platform does not satisfy this requirement

Why this is production-relevant beyond the model itself:

- **Portability:** the same container image and the same trained artifacts run unmodified on your laptop, inside Docker, and on a managed cloud host.
- **Consistency:** no "but it works on my machine" gap between what you demo locally and what a domain expert sees on a public URL.
- **Separation of concerns:** a REST API (Render), a no-code demo UI (Hugging Face/Streamlit), and your own frontend (the Advanced-tier HTML pages) are three different ways of exposing the *same* underlying models to three different kinds of user -- a developer integrating your API, a non-technical domain expert validating it, and an end customer.

---

## Marking Guide

Refer to the marking guide available here: [lab_submission_marking_guide.md](lab_submission_marking_guide.md)