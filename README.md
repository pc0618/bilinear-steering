# bilinear-steering

This work was part of a CS 224N (NLP with Deep Learning at Stanford) Final Project submission. A link to the final project report can be found here.
https://drive.google.com/file/d/1ms_4Tlka_ebJcWExxgISF4E98xOiCacR/view?usp=sharing


While many recent works propose steering methods for Transformer-based lan-
guage models to align outputs with human preferences and values, the efficacy of
techniques beyond prompt engineering remains uncertain. This reflects a broader
gap in mechanistic understanding of how language models encode and utilize abstract concepts. In this work, we investigate whether a recently proposed weight-
based approach to mechanistic interpretability [1] can be leveraged for modelsteering. Specifically, we pretrain a modified TinyLlama-1.1B model incorporating
bilinear MLP layers, train linear probes on MLP block outputs to predict positive
sentiment, and attempt to steer the model away from positive sentiments by adding
regularizer terms to the loss in a post-training round to reduce the contributions
from output directions identified by linear probes. Our findings show that there
is further work to be done to bridge the gap between model interpretability and
steerability, providing insights on the potential of weight-based interventions for
steering language models.
