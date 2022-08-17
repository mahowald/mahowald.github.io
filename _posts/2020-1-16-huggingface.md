---
layout: post
title: "Getting started with Huggingface Transformers"
tagline: "Robots in disguise"
tags: [data science]
author: Matthew Mahowald
mathjax: true
---

Everyone's favorite open-source NLP team, Huggingface, maintains a library ([Transformers](https://github.com/huggingface/transformers)) of PyTorch and Tensorflow implementations of a number of bleeding edge NLP models.
I recently decided to take this library for a spin to see how easy it was to replicate [ALBERT's](https://arxiv.org/abs/1909.11942) performance on the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/).
At the time of writing, the documentation for this package is under active development, so the purpose of this post is mostly to capture a few things that weren't immediately obvious---namely, how to use `Pipelines`.

Getting started
---------------

For convenience and reproducibility, I decided to package my code as an executable module and containerize it.
I'm using the PyTorch implementations of the models Huggingface provides, and I decided to name my module `squadster`.
My Dockerfile looks like this:

```Dockerfile
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

WORKDIR /src/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY setup.py .
COPY squadster/ ./squadster/
RUN pip install .

ENTRYPOINT ["python", "-m", "squadster"]
```

Huggingface added support for pipelines in v2.3.0 of Transformers, which makes executing a pre-trained model quite straightforward.
For example, to use ALBERT in a question-and-answer pipeline only takes two lines of Python:
```python
from transformers import pipeline

nlp = pipeline(task="question-answering", model="albert-large-v2")
```

To make this a self-contained question-answering system we can query, I've created the alias function `predict`:
```python
def predict(model, context, question):
    return model({"question": question, "context": context})
```
and hooked it up to `stdin` and `stdout` in my `__main__.py` file:

```python
# __main__.py

import click
import torch
import sys
import json

from transformers import pipeline
from .predict import predict


@click.command()
@click.option("--mode", default="predict")
def main(mode):
    sys.stderr.write("{}".format(torch.cuda.get_device_name(0)))
    context = 'YOUR CONTEXT HERE!'

    sys.stderr.write("Context: {}".format(context))

    nlp = pipeline(task="question-answering", model="albert-large-v2")
    question = sys.stdin.read()
    result = predict(nlp, context, question)

    sys.stdout.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
```

Asking about Normandy
---------------------

You can download some sample questions and responses from the SQuAD website.
For example, one of the first contexts provided is

> The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'

with the proposed question "In what country is Normandy located?".

Unfortunately, `albert-large-v2` gets this one wrong:
```
echo "In what country is Normandy located?" | \
docker run -i --gpus all local/squad
```
returns
```
{
    "score": 3.239981109320325e-05, 
    "start": 32, 
    "end": 49, 
    "answer": "French: Normands;"
}
```

Note as well that even small variations in the question can also produce different responses---asking "What country is Normandy located in?" produces the response
```
{
    "score": 5.0038562250316696e-05, 
    "start": 451, 
    "end": 501, 
    "answer": "Roman-Gaulish populations, their descendants would"
}
```

(A full list of the [available model aliases can be found here.](https://huggingface.co/transformers/pretrained_models.html))

Model training
--------------

I expected to write more about model training, but Huggingface has actually made it super easy to fine-tune their model implementations---for example, [see the `run_squad.py` script](https://github.com/huggingface/transformers#run_squadpy-fine-tuning-on-squad-for-question-answering).
This script will store model checkpoints and predictions to the `--output_dir` argument, and these outputs can then be reloaded into a pipeline as needed using the `from_pretrained()` methods, for example:

```python
p = pipeline('question-answering', 
             model=AutoModel.from_pretrained(...), 
             tokenizer=AutoTokenizer.from_pretrained(...),
            )
```

Note that in practice fine-tuning models can be quite demanding---for example, the smaller BERT base models have about 110 million parameters.
(A list of the [supported pretrained models can be found here](https://huggingface.co/transformers/pretrained_models.html))

Anyway, that's all for now.
I'll update this post as I delve further!



