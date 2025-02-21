# CS 224N Default Final Project: Build GPT-2

Authors: Elena Sierra & Anika Fuloria

## VERSE: Volta-Embedded Refined Sonnet Engine

### Goal
Our research investigates whether we can improve upon the contributions of *The Mechanical Bard* by generating Shakespearean sonnets that feel more artistic and less robotic. Specifically, we aim to determine if large language models (LLMs), such as GPT-2, can be guided to produce sonnets that exhibit a *volta*, or emotional shift, in accordance with Shakespearean convention. By incorporating explicit probabilistic sentiment changes, we hypothesize that our generated sonnets will adhere to the expected tonal structure of Shakespearean poetry, enhancing their literary authenticity.

As AI-generated text becomes more prevalent in creative writing and everyday applications, developing methodologies to imbue generated content with "humanity" is essential. *The Mechanical Bard* demonstrated that rule-based constraints improve grammatical correctness and poetic structure, yet its output lacks the depth of a poet's intent; there is little attention to conceit (extended metaphor) or tonal evolution. Our work seeks to address this gap by explicitly modeling the volta within generated text while maintaining the structural advantages of *The Mechanical Bard*’s methodology.

Our primary goal is to generate sonnets that maintain Shakespearean rhyme and meter while exhibiting clear emotional shifts at conventional positions (e.g., lines 8-9 or 12-13). If time permits, we aim to extend our model to incorporate an overarching conceit, enhancing thematic unity within the poem.

### Task
We will generate sonnets using a constrained GPT-2 model, similar to *The Mechanical Bard*, but with explicit modeling of tonal shifts. The model should:
- Generate 14-line sonnets following the Shakespearean rhyme scheme (ABAB CDCD EFEF GG).
- Maintain iambic pentameter.
- Introduce a distinct volta at line 8-9 (Petrarchan-style) and/or line 12-13 (Shakespearean-style), reflecting an emotional or argumentative shift.

#### Example Input and Output:
- **Input:** "Theme: Love"
- **Output:** A sonnet that describes love’s beauty in the quatrains, then suddenly shifts to the loss of a lover and reminiscing over her through poetry.
  
  Example from Shakespeare’s *Sonnet 18*:
  
  ```
  Shall I compare thee to a summer’s day?
  Thou art more lovely and more temperate:
  Rough winds do shake the darling buds of May,
  And summer’s lease hath all too short a date;
  Sometime too hot the eye of heaven shines,
  And often is his gold complexion dimm’d,
  And every fair from fair sometime declines,
  By chance or nature’s changing course untrimm’d:
  But thy eternal summer shall not fade,
  Nor lose possession of that fair thou ow’st,
  Nor shall Death brag thou wand’rest in his shade,
  When in eternal lines to time thou grow’st.
  So long as men can breathe or eyes can see,
  So long lives this, and this gives life to thee.
  ```
  
  There is a clear volta between lines 12 and 13, when the speaker shifts from praising his lover's beauty to her immortal representation in his literary works.

### Data
We will use the following datasets:
- The Shakespearean sonnets dataset provided by Stanford.
- Additional sonnets from poets who followed traditional Shakespearean structures, such as Edna St. Vincent Millay and W.H. Auden.

Preprocessing will involve tokenization, POS tagging, and sentiment labeling at the line level to analyze the presence of volta in training data.

### Methods
Our approach builds upon *The Mechanical Bard* but introduces key innovations:
- We will use a similar template-based approach for grammatical correctness and rhyme adherence.
- In addition to the template-based approach implemented in *The Mechanical Bard*, we will introduce an explicit probabilistic sentiment shift mechanism to enforce volta placement.
- Our generation process will:
  1. Preselect rhyming words and establish a POS template.
  2. Generate quatrains with a controlled sentiment trajectory.
  3. At predefined volta positions (lines 8-9 or 12-13), introduce a shift in sentiment using probabilistic modeling, ensuring a contrast in tone.
  4. Use a constrained decoding method (e.g., beam search with filtering) to ensure adherence to structure.

### Baselines
We will compare our approach against the following baselines:
1. GPT-2 generating sonnets without structure constraints (baseline from *The Mechanical Bard*).
2. GPT-2 following explicit structure constraints but without sentiment control (contributions from *The Mechanical Bard*).
3. Our new model incorporating probabilistic volta placement.
4. (Stretch goal) Our model incorporating volta + an overarching conceit for thematic unity.

### Evaluation
We will use both quantitative and qualitative evaluation metrics:
- **Cross-entropy loss**: Measures adherence to expected probability distributions during text generation.
- **Poetic quality assessment**: Following *The Mechanical Bard*, we will conduct human evaluations with English literature students and other Stanford students to assess whether the generated sonnets:
  - Adhere to rhyme/meter constraints.
  - Exhibit natural language fluency.
  - Successfully incorporate a volta.
  - Appear human-authored ("Turing Test" for poetry).
- **Sentiment analysis**: We will evaluate sentiment shifts before and after volta positions using a sentiment classifier to ensure the expected tonal contrast is present.

### Ethical Considerations
One ethical challenge of this project is the potential for AI-generated poetry to blur the line between human and machine authorship, raising concerns about intellectual property and artistic authenticity. If AI-generated sonnets are indistinguishable from human-written ones, there is a risk that AI could be misused to generate works falsely attributed to human poets, undermining the value of human creativity. To mitigate these risks, we propose transparency measures such as clear labeling of AI-generated sonnets to distinguish them from human-authored poetry. We will also only train on poetry in the public domain, to avoid issues of copyright and ownership.

Another societal risk is the potential reinforcement of biased or stereotypical language patterns, especially given that the training data consists primarily of historical texts. Shakespearean sonnets, while widely admired, reflect the social norms and biases of their time, and an AI model trained on such works might unintentionally perpetuate outdated or exclusionary perspectives. To address this, we propose adding new sonnets to our training set from authors of diverse backgrounds. We will also audit our outputs to make sure that offensive content is not generated.

### Conclusion
By enforcing sentiment shifts within a constrained generation framework, we aim to enhance the artistic quality of AI-generated sonnets, making them more reflective of Shakespearean poetic tradition. If successful, our approach could serve as a blueprint for improving the expressiveness of AI in creative writing applications.

## Default project instructions

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course
website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the GPT-2 model to
better understand its architecture.
In the second part, you will use the token embeddings produced by your GPT-2 model on two downstream tasks: paraphrase
detection and sonnet generation. You will implement extensions to improve your model's performance on these tasks.

In broad strokes, Part 1 of this project targets:

* modules/attention.py: Missing code blocks.
* modules/gpt2_layer.py: Missing code blocks.
* models/gpt2.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

To test Part 1, you will run:

* `optimizer_test.py`: To test your implementation of `optimizer.py`.
* `sanity_check.py`: To test your implementation of GPT models.
* `classifier.py` : To perform sentiment classification using your models.

In Part 2 of this project, you will use GPT2 (via cloze-style classification) detect if one sentence is a paraphrase of 
another as well as generate sonnets via autoregressive language modeling.  

To test Part 2, you will run:

* `paraphrase_detection.py`: To perform paraphrase detection. 
* `sonnet_generation.py`: To perform sonnet generation.

Important: Adjust training hyperparameters, particularly batch size, according to your GPU's specifications to optimize performance and prevent out-of-memory errors.

## Pre-testing instructions

While there are missing code blocks that you need to implement in both of these files, the main focus of this second 
part are the extensions: how you modify your GPT2 model to improve its ability to determine if one sentence is a 
paraphrase of another as well as its ability to generate sonnets. 

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).

Inspiration for this project from *The Mechanical Bard* (https://aclanthology.org/2023.acl-short.140/)