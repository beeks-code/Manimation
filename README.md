#  Manimation

This repository contains **high-quality animations of Transformer concepts** built using [Manim](https://www.manim.community/).

This code was featured in a video I made about the Self-Attention Mechanism.

**Video Link:** https://youtu.be/ZDQZnb6Z350?si=uNpVk51XUFITgorG

The goal is to visually explain:
- Self-Attention mechanism
- Query, Key, Value (Q, K, V)
- Matrix multiplication (QKᵀ)
- Scaling and Softmax
- Contextual embeddings

  
---

#  Requirements

- Python 3.8+
- Manim Community Edition 
#  Installation
 - pip install manim
## To Run
```bash
git clone https://github.com/beeks-code/Manimation.git
cd "Manimation/Attention Mechanism"

Medium Quality
manim -pqm file_name.py SceneName

High Quality (Full HD)
manim -pqh file_name.py SceneName

4K Quality (Ultra HD)
manim -pqk file_name.py SceneName



