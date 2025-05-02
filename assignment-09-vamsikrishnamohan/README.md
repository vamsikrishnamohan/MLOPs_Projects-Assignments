[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tKoQrp-x)

# DA24M026
# Vamsi krishna Mohan.S 

---

##  Problem Statement

The objective of this assignment is to apply **distributed computing and sentiment analysis** techniques on a real-world dataset using the **MapReduce paradigm**.

We use a large **Amazon Gourmet Foods reviews dataset** and:

- Perform **sentiment classification** of each review text using a **pretrained HuggingFace model**.
- Compare the predicted sentiment with the actual **review star rating** (considering rating ≥ 3.0 as POSITIVE).
- Compute **Precision**, **Recall**, and **Confusion Matrix** to evaluate the model.
- Use **Ray** for efficient, scalable **parallel processing** across CPU cores.

---

##  Tasks Overview

###  Task 1:  
Use a **pretrained transformer pipeline** for sentiment analysis and run it on every record in a distributed manner.  
Each record is labeled as either **POSITIVE** or **NEGATIVE** based on the `review/text`.

###  Task 2:  
Use the ground-truth `review/score` to generate actual sentiment labels.  
Compute:
- Confusion Matrix
- Precision
- Recall

---

##  Setup Instructions

1. **Clone the repo**:

```bash
git clone https://github.com/DA5402-MLOps-JanMay2025/assignment-09-vamsikrishnamohan.git
cd assignment-09-vamsikrishnamohan
```

2. **Install required packages**:

```bash
pip install -r requirements.txt
```

3. **Add dataset**:  
Place the `Gourmet_Foods.txt` file in the root directory (not tracked in Git, see `.gitignore`).

---

##  Running the Script

```bash
python main.py
```

This will:
- Spawn multiple Ray workers
- Run sentiment classification in parallel
- Compute and display the confusion matrix, precision, and recall
- Show a heatmap of the confusion matrix

---

##  Output Example

```
Confusion Matrix:
 [[107921  25812]
  [ 2181  18721]]

Precision: 0.9802
Recall:    0.8070
```

A visual confusion matrix will also be shown using `seaborn`.

---

##  File Structure

```
├── main.py              # Main script for distributed processing
├── requirements.txt     # All required dependencies
├── .gitignore           # Ignore dataset and environment files
├── README.md            # Assignment report
```

---

##  What is Ray?

**Ray** is a powerful Python framework for building distributed applications. It allows you to run Python functions in **parallel across all CPU cores or machines** — with very little code change.

###  Why Ray for This Assignment?

- Ray makes it **easy to run sentiment analysis on hundreds of reviews at once**.
- It’s **more stable than joblib or multiprocessing** when using large ML models like HuggingFace transformers.
- Each review is processed using a `@ray.remote` function, and results are gathered using `ray.get()`.

###  How It Works

```python
@ray.remote
def process_review(review):
    # load model
    # return sentiment prediction

futures = [process_review.remote(r) for r in reviews]
results = ray.get(futures)
```

This lets you scale easily without worrying about process or thread management.

### 💡 Key Features

|RAY|
|--------|
| Easy parallelism |
| Works with HuggingFace models |
| Scales to clusters |
| Built-in fault tolerance |

---
