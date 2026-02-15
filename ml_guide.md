# Hack The East - Machine Learning Guide

Author: George Huang, Software Engineer @ SixMac (Shanghai)

- My Linkedin: [Linkedin profile](https://www.linkedin.com/in/george-huang-535109269/)
- My GitHub: [GitHub profile](https://github.com/Goge052215)

---

## Intro

This guide should be read after you guys finish the first guide: "Everything You need for Hack The East", as this guide covers the basics of Machine Learning.

Related topics: data preprocessing, logistic regression, confusion matrix, and Next.js workflow.

I assume you guys have taken, are taking, or will take **STAT2602**, **STAT3600**, and **STAT3612** in the future, as this guide contains some **maths**.

### Background Information

In real world, we have tasks that require us to judge or classify cases into several criteria, making workflow and decisions easier. For instance,

- We might determine a patient's symptom based on their information. This dataset is the famous Breast Cancer Dataset*.
- We can identify the flower type of a flower based on its petal colors, size, growth pattern, etc. This is the famous Iris Dataset*.

Both examples above are popular for data scientists and students to study *Machine Learning*, specifically the **Classification Task**. Machine learning has many fields, but classification remains one of the most *classic* and *prevailing* domains globally.

For more information about Machine Learning, here are some additional background for you guys to delve into:

- []()
- []()

---

## Classification

### Introduction to Classification

Qualitative variables take values in an unordered set $\mathcal{C}$, such as:

- $\text{eye color} \in \{\text{brown}, \text{blue}, \text{green}\}$
- $\text{email} \in \{\text{spam}, \text{ham}\}$

They are also referred to as categorical.

Given a feature vector $X$ and a qualitative response 𝑌 taking values in the set $\mathcal{C}$, the classification task is to build a function (or model, classifier) $C(X)$ that takes as input the feature vector $X$ and predicts its value for $Y$ ; i.e., $C(X) \in \mathcal{C}$.

We are more interested in estimating the probabilities that $X$ belongs to each category in $\mathcal{C}$. For example, it is more valuable to have an estimate of the probability that an
insurance claim is fraudulent, than a classification fraudulent or not.

### Logistic Regression (Multinomial)

#### Scenario

As mentioned, the core idea of classification is estimating the probability of $Y$ belonging to each class. In our Mindex project, we have **three** sentiment classes:

$$Y = \begin{cases} 0 &\text{if Neutral}; \\ 1  &\text{if Negative}; \\ 2 &\text{if Positive}. \end{cases}$$

Since we have more than two categories ($K=3$), simple binary logistic regression is generalized into **Multinomial Logistic Regression** (also known as **Softmax Regression**).

#### From Text to Numbers (TF-IDF)

Before we can use any math, we must convert our input text (tweets) into numbers. We use **TF-IDF** (Term Frequency-Inverse Document Frequency).

For a given word $t$ in a document $d$:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

This gives us a large feature vector $X$ for each tweet, where each element represents the importance of a word. In our code, we limit this to the top 20,000 features (`max_features=20000`) and include 2-word phrases (`ngram_range=(1, 2)`).

#### The Softmax Function

In binary logistic regression, we used the **Sigmoid** function to squash a value between 0 and 1. For multi-class problems, we use the **Softmax** function.

For each class $k \in \{0, 1, 2\}$, the model calculates a "score" (logit) $z_k$:

$$z_k = \beta_{k0} + \beta_{k1}X_1 + \dots + \beta_{kp}X_p$$

Then, the probability that the observation belongs to class $k$ is given by:

$$\Pr(Y=k \mid X) = \frac{e^{z_k}}{\sum_{j=0}^{K-1} e^{z_j}}$$

This ensures that the probabilities for all classes sum up to 1:

$$\sum_{k=0}^{2} \Pr(Y=k \mid X) = 1$$

#### Maximum Likelihood Estimation (Cross-Entropy Loss)

To train the model (find the best $\beta$ values), we maximize the likelihood of the observed data. In machine learning, this is equivalent to minimizing the **Cross-Entropy Loss**.

For a single data point with true class $y$ (one-hot encoded) and predicted probabilities $\hat{y}$:

$$L(\beta) = - \sum_{k=0}^{K-1} y_k \log(\hat{y}_k)$$

Our training process finds the parameters $\beta$ that minimize this loss across all training examples (27,481 tweets).

---

## Demo Project - Mindex

### What is Mindex?

'Mindex' is a platform where we input a text, then let the model analyze it and return the sentiment.

It has a workflow of:

```
1. Input text
2. Next.js sends the text to the Python model
3. Python model runs and returns results
4. Output results
```

Results include:

- Overall sentimentality: Positive / Neutral / Negative
- Sentimentality type distribution, for instance:
	- 82% neutral
	- 13% positive
	- 5% negative

### Project Setup

Now let's start building the application! We will set up a Next.js environment for the frontend and a Python environment for the machine learning model.

#### 1. Initialize Next.js App

First, create the Next.js project with the necessary flags:

```bash
mkdir mindex
npx create-next-app@latest . --typescript --tailwind --eslint --yes
cd mindex
```

#### 2. Create Model Folder

Create a directory `src/` for the core modelling:

```bash
mkdir src
```

We will eventually populate this directory with:
- `main.py`: The entry point for prediction
- `preprocessing.py`: Data cleaning and transformation
- `trainer.py`: Model training pipeline

#### 3. Python Virtual Environment

It is best practice to use a virtual environment for dependencies. Run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install pandas scikit-learn numpy matplotlib seaborn
```

At this stage, your project structure should look like this:

```
├── src/        <-- Currently empty (will hold model code)
├── venv/       <-- Python virtual environment
├── package.json
├── ...
```

### Dataset & Preprocessing

#### Dataset

Our [Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/) is obtained from Kaggle, by @abhi8923shriv. The training dataset has a size of:

- 10 features: `textID`, `text`, `selected_text`, `sentiment`, `Time of Tweet`, `Age of User`, `Country`, `Population -2020`, `Land Area (Km^2)`, `Density (P/Km^2)`.
- 27481 lines of data, meaning that we have 27481 data points for us to train.

The testing dataset has a size of:

- 9 features: `textID`, `text`, `sentiment`, `Time of Tweet`, `Age of User`, `Country`, `Population -2020`, `Land Area (Km^2)`, `Density (P/Km^2)`. Test set is missing `selected_text`.
- 3535 lines of data, meaning that test set size is around **12%** of the training set.

Download the CSVs and put them in `data/`. First, create the directory:

```bash
cd mindex && mkdir data
```

#### Processing

For preprocessing, we follow these steps:

1. drop columns that are not useful
2. encode the categorical variables
3. drop rows with missing values

Here is a complete workflow of data preprocessing:

```python
# preprocessing.py

import pandas as pd
class Preprocessor:
    def __init__(self, df):
        self.df = df
    
    # Drop columns that are not useful (ID and raw text)
    def drop_unuseful_columns(self):
        self.df = self.df.drop(['textID', 'text', 'selected_text'], axis=1, errors='ignore')
    
    # Encode the categorical variables
    def encode_categorical_variables(self):
        self.df['sentiment'] = self.df['sentiment'].map({
            'neutral': 0, 
            'negative': 1, 
            'positive': 2
        })
        
        # Encode other categorical columns
        categorical_cols = ['Time of Tweet', 'Age of User', 'Country']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category').cat.codes

    # Preprocess the data
    def preprocess(self):
        self.drop_unuseful_columns()
        self.encode_categorical_variables()
        self.df = self.df.dropna()
```

Note that, here we drop the rows not only with missing columns, but also columns that has lower potential in sentimentality analysis, such as `textID`, `text`, `selected_text`.

### Model Training

I have finished the building part for you guys. You can directly copy and paste the code below into your project, but **please make sure you understand what I am doing** (check the comments and the explanation above).

We use a **Pipeline** to combine the vectorizer and the classifier into a single model object. This makes it easy to train and predict without handling the text transformation separately every time.

```python
# trainer.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
import preprocessing as pp

class Trainer:
    def __init__(self, train_df):
        self.raw_train_df = train_df
        self.preprocessor = pp.Preprocessor(train_df)
        self.preprocessor.preprocess()

    def _prepare_text_df(self, df):
        df = df[['text', 'sentiment']].copy()
        df['sentiment'] = df['sentiment'].map({
            'neutral': 0,
            'negative': 1,
            'positive': 2
        })
        df = df.dropna()
        return df

    def train_text(self, test_df):
        train_text_df = self._prepare_text_df(self.raw_train_df)
        test_text_df = self._prepare_text_df(test_df)

        X_train = train_text_df['text']
        y_train = train_text_df['sentiment']
        X_test = test_text_df['text']
        self.y_test = test_text_df['sentiment']

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("logreg", LogisticRegression(max_iter=1000))
        ])
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)

        result = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred, average='weighted'),
            "f1": f1_score(self.y_test, self.y_pred, average='weighted'),
            "recall": recall_score(self.y_test, self.y_pred, average='weighted'),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred).tolist(),
            "classification_report": classification_report(self.y_test, self.y_pred)
        }

        return result

    def train(self, test_df, model_type="tfidf_logreg"):
        return self.train_text(test_df)
    
    # confusion matrix for testing and model dev
    def conf_matrix(self):
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            confusion_matrix(self.y_test, self.y_pred), 
            annot=True, fmt='d', cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def save_model(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
```

The `train.py` script orchestrates the loading of data, initializing the trainer, and saving the final model to a file (`sentiment_model.pkl`) so it can be reused by the website.

```python
# train.py

import os
import pandas as pd
import trainer

def train():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    model_path = os.path.join(base_dir, 'src', 'sentiment_model.pkl')

    print(f"Loading data from {data_dir}...")
    try:
        train_df = pd.read_csv(train_path, encoding='latin1')
        test_df = pd.read_csv(test_path, encoding='latin1')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Initializing trainer...")
    t = trainer.Trainer(train_df)
    
    print("Training model...")
    metrics = t.train(test_df)
    print("Training complete. Metrics:")
    print(metrics)
    
    print(f"Saving model to {model_path}...")
    t.save_model(model_path)
    print("Done.")

if __name__ == "__main__":
    train()
```

Now run `train.py` to train the model. It saves a pickle file `sentiment_model.pkl` in `src/` so we can reuse the trained model without retraining every time. This makes prediction fast and consistent.

### Prediction (Inference)

When a user types text on the website, `main.py` loads the saved model and predicts the sentiment. We also calculate the probability for each class to show the distribution.

```python
# main.py

import sys
import os
import pickle
import json
import warnings

# Suppress sklearn warnings about pickling
warnings.filterwarnings("ignore")

def predict(text):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'src', 'sentiment_model.pkl')
    
    if not os.path.exists(model_path):
        return {
			"status": "error", 
			"message": "Model not found. Please run src/train.py first."
		}
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Predict
        # Model expects an iterable of strings
        prediction_code = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        # Map back to string
        # 0: neutral, 1: negative, 2: positive (from trainer.py)
        sentiment_map = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}
        sentiment = sentiment_map.get(prediction_code, "Unknown")
        
        # Format probabilities
        # classes_ are usually sorted as [0, 1, 2] corresponding to Neutral, Negative, Positive
        # We need to map them correctly based on model.classes_
        
        probs_dict = {}
        for idx, class_label in enumerate(model.classes_):
            label_name = sentiment_map.get(class_label, str(class_label))
            probs_dict[label_name] = round(probabilities[idx] * 100, 1)

        return {
            "status": "success", 
            "data": sentiment,
            "distribution": probs_dict
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
			"status": "error", 
			"message": "No input text provided"
		}))
        sys.exit(1)
        
    text = sys.argv[1]
    result = predict(text)
    print(json.dumps(result))
```

### Result

Now you are all done! The modeling file `src/` should have a structure of:

```
├── data/
│   ├── test.csv             <-- testing dataset
│   └── train.csv            <-- training dataset
├── src/
│   ├── main.py              <-- main execution script
│   ├── preprocessing.py     <-- dataset preprocessing
│   ├── sentiment_model.pkl  <-- saved model file
│   ├── train.py             <-- training script
│   └── trainer.py           <-- model training file
```

where `data/` is the directory of the training and testing set, and `src/` is the directory of core modelling. 

If you guys are lost when building the app, do **double check** with this file structure again! This structure is *proven* when I ran them before this guide's releasement.

After finishing this part, you should have an understanding of how the Machine Learning model works and how to use it for prediction. In fact, I do conduct the testing result for my model:

- **Accuracy**: 70.4%
- **Precision**: 71.5%
- **F1 Score**: 70.5%
- **Recall**: 70.4%

Confusion Matrix:

| True \ Pred | Neutral | Negative | Positive |
| :---------: | :-----: | :------: | :------: |
| **Neutral** | **1082** |   187   |    161   |
| **Negative** |   343  |  **634** |    24    |
| **Positive** |   287  |    43    |  **773** |

Overall, the model performs well with an accuracy of 70.4%, precision of 71.5%, F1 score of 70.5%, and recall of 70.4%. The confusion matrix shows that the model is *particularly good* at classifying positive sentiments but struggles with negative sentiments (for the baseline, of course).

Given this is a demo project, I guess I will stop the modelling here and start shipping the website!

---

### Next.js Frontend Implementation

The structure looks the same as the previous calculator one, with the code content might be different:

```
├── data/
│   ├── train.csv           <-- training dataset
│   └── test.csv            <-- testing dataset
├── app/
│   ├── page.js             <-- The main UI
│   ├── globals.css
│   ├── layout.tsx
│   ├── ThemeContext.js     <-- Theme Context
│   ├── api/
│   │   └── route.js    <-- The "Bridge"
│   └── settings/           <-- Settings Page
│       └── page.js
├── src/
│   ├── main.py              <-- main execution script
│   ├── preprocessing.py     <-- dataset preprocessing
│   ├── sentiment_model.pkl  <-- saved model file
│   ├── train.py             <-- training script
│   └── trainer.py           <-- model training file
└── venv/                    <-- virtual environment
```

Again, this serves as a double-check for the file structure. If you are stuck at any point, double-check this file structure (or DM me).

#### Page UI (Essential Only)

Here is the essential UI logic that connects input to the API, supports batch input, and keeps a small local history. For the full code, please check the [repo HERE](https://github.com/goge052215/mindex).

```javascript
'use client';
import { useState, useEffect } from 'react';

export default function SentimentAnalyzer() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');
  const [distribution, setDistribution] = useState(null);
  const [history, setHistory] = useState([]);
  const [batchResults, setBatchResults] = useState([]);

  useEffect(() => {
    const savedHistory = localStorage.getItem('history');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('history', JSON.stringify(history));
  }, [history]);

  const handleExecute = async () => {
    if (!input) return;

    const lines = input
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length === 0) return;

    const payload = lines.length > 1 ? lines : lines[0];
    const res = await fetch('/api', {
      method: 'POST',
      body: JSON.stringify({ expression: payload })
    });
    const data = await res.json();

    if (data.status === 'success') {
      if (Array.isArray(data.data)) {
        setBatchResults(data.data);
        setResult('');
        setDistribution(null);
        setHistory((prev) => [...data.data, ...prev].slice(0, 10));
      } else {
        setResult(data.data);
        setDistribution(data.distribution);
        setBatchResults([]);
        setHistory((prev) => [{ text: lines[0], sentiment: data.data, distribution: data.distribution }, ...prev].slice(0, 10));
      }
    } else {
      setResult('Error');
      setDistribution(null);
      setBatchResults([]);
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return (
    <div>
      <textarea value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={handleExecute}>Analyze Sentiment</button>
      {result && <div>{result}</div>}
      {distribution && (
        <div>
          {Object.entries(distribution).map(([label, score]) => (
            <div key={label}>{label}: {score}%</div>
          ))}
        </div>
      )}
      {batchResults.length > 0 && (
        <div>
          {batchResults.map((entry, idx) => (
            <div key={idx}>{entry.text} → {entry.sentiment}</div>
          ))}
        </div>
      )}
      {history.length > 0 && (
        <div>
          <button onClick={clearHistory}>Clear History</button>
          {history.map((entry, idx) => (
            <div key={idx}>{entry.text} → {entry.sentiment}</div>
          ))}
        </div>
      )}
    </div>
  );
}
```

#### API Route (Essential Only)

This API route runs the Python model, supports batch input, and caches recent results.

I implemented a simple LRU cache with a max size of 500 and a TTL of 5 minutes. LRU means “least recently used”: when the cache is full, we drop the oldest unused entry. This makes repeated inputs return instantly while keeping memory bounded. The TTL ensures entries expire even if they are not evicted by size.

For a detailed explaination of LRU cache, please check [this article](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU).

```javascript
import { spawn } from 'child_process';
import path from 'path';

const CACHE_MAX = 500;
const CACHE_TTL_MS = 5 * 60 * 1000;
const cache = new Map();

function getCached(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (entry.expiresAt <= Date.now()) {
    cache.delete(key);
    return null;
  }
  cache.delete(key);
  cache.set(key, entry);
  return entry.value;
}

function setCached(key, value) {
  cache.set(key, { value, expiresAt: Date.now() + CACHE_TTL_MS });
  while (cache.size > CACHE_MAX) {
    const oldestKey = cache.keys().next().value;
    cache.delete(oldestKey);
  }
}

async function trySpawn(command, args) {
  return new Promise((resolve) => {
    const process = spawn(command, args);
    let result = '';
    let error = '';

    process.stdout.on('data', (data) => { result += data.toString(); });
    process.stderr.on('data', (data) => { error += data.toString(); });

    process.on('error', (err) => {
      resolve({ success: false, error: err.message });
    });

    process.on('close', (code) => {
      if (code === 0) {
        resolve({ success: true, data: result });
      } else {
        resolve({ success: false, error: error || `Exit code ${code}` });
      }
    });
  });
}

export async function POST(req) {
  const { expression } = await req.json();
  const scriptPath = path.join(process.cwd(), 'src/main.py');
  const payload = typeof expression === 'string' ? expression : JSON.stringify(expression);
  const cached = getCached(payload);
  if (cached) {
    return new Response(cached, {
      headers: { 'Content-Type': 'application/json' },
    });
  }
  const commands = [
    path.join(process.cwd(), 'venv/bin/python'),
    'python', 'python3', 'py'
  ];
  let lastError = '';

  for (const cmd of commands) {
    const outcome = await trySpawn(cmd, [scriptPath, payload]);
    if (outcome.success) {
      setCached(payload, outcome.data);
      return new Response(outcome.data, {
        headers: { 'Content-Type': 'application/json' },
      });
    }
    lastError = outcome.error;
    if (!lastError.includes('ENOENT') && !lastError.includes('not found') && !lastError.includes('9009')) {
      break;
    }
  }

  return new Response(JSON.stringify({ 
    status: 'error', 
    message: `Failed to parse the text. Python error: ${lastError}` 
  }), {
    status: 500,
    headers: { 'Content-Type': 'application/json' },
  });
}
```

For styling, theming, and settings UI, use the complete code from the repo.

#### Layout Typescript

Last but not least, `layout.tsx` is the layout file of the app, it is charged for the layout of the app.

```typescript
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "./ThemeContext";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
        suppressHydrationWarning={true}
      >
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
```

And you are done! Now, just run the app

```bash
npm run dev
```

and open `http://localhost:3000` in your browser. You should see a beautiful sentiment analysis webapp. Try with it by typing some sentences and feel the power of Machine Learning!

### GitHub Repository

You might notice that I have omitted some code in the codeblock above. Do not panic! I have put the complete code in the repo. You can check it out by clicking the link below.

[Mindex Repository](https://github.com/Goge052215/mindex)

Just follow the guide above and the complete code in the github repo, and you will have a working sentiment analysis webapp as I shown in the group chat!

Now, I will introduce on how to deploy the app on Vercel.

---

## Vercel

### What is Vercel?

Vercel is a cloud platform for static sites and Serverless Functions. It makes deployment and scaling simple, and it works great with Next.js.

Why we use Vercel?

- Easy to deploy
- Scalable by default
- Global CDN

### Setup (Brief)

0. Check the repo for this 'Mindex' project.
1. Copy the code to your own project file.
2. Start a new git repo for your project, and connect with GitHub.
3. Go to https://vercel.com and import the repo.
4. Select **Next.js** as the framework and keep the default build settings.
5. Click **Deploy** and wait for the live URL.

Note: If your API depends on local Python execution, host the Python model separately and call it from the Next.js API route.

### More Information

Here are some more resources to help you understand the concepts better. As I said, I will explain more about Vercel in the workshop after CNY.

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Documentation](https://nextjs.org/docs)

---

## Next Steps - Checkpoint before We Leave

So far, you should have some solid understanding of

- [ ] Logistic Regression
- [ ] Model Training Pipeline
- [ ] Model Evaluation & Saving Trained Model
- [ ] Webapp Deployment

Make sure you have completed all the steps above. Again, the topics I have covered are important and extreme suitable for a starter. I suggest spend 2-4 hours to go through them, and you will be EVEN MORE comfy with the workshop.

Happy coding!

George Huang,

Feb. 14, 2026
