# Understanding S-JEPA, step by step (English)

Hello and welcome! This guide explains S-JEPA from zero. No heavy maths, lots of
small steps and analogies. If you have never trained a speech model before, this
is for you. Take your time and read slowly.

> **What you will learn.** What problem S-JEPA solves, what each part of the
> model does, and how the training really works under the hood.

---

## 1. The big picture

Imagine you want a computer to "understand" speech. The classic way needs a lot
of labeled data: someone must write down what is said in every clip. That is
slow and expensive.

**Self-supervised learning** is a trick to learn *without labels*. The model
plays a game with itself: we hide a part of the sound and ask the model to guess
what was hidden. To win the game, the model must learn the structure of speech.

> *Naive question: "Guess the hidden part? Like filling a missing word in a
> sentence?"*
>
> Yes, exactly. But here the "words" are tiny 20-millisecond sound frames, and
> the model does not guess the raw sound. It guesses a **soft label** for each
> frame. We will see what a soft label is very soon.

---

## 2. The frames: cutting sound into small pieces

Sound is a long wave. We cut it into small frames of 20 ms each. At 16 kHz, one
frame is 320 samples. A 10-second clip becomes about 500 frames.

A small **CNN frontend** does this cutting and turns each frame into a vector of
numbers. Think of it as a machine that reads the wave and writes one "summary
card" per frame.

```
raw wave  ->  [CNN frontend]  ->  frame 1, frame 2, frame 3, ... frame T
```

---

## 3. The three trained parts

S-JEPA has three parts that learn together:

| Part          | Symbol     | Job                                              |
|---------------|------------|--------------------------------------------------|
| Encoder       | `f_phi`    | Read the frames and build rich representations.  |
| Predictor     | `h_psi`    | Fill in the hidden frames from the visible ones. |
| Cluster head  | `g_omega`  | Turn a frame into `K` cluster scores (logits).   |

> **Analogy.** The encoder is a *reader* who understands the context. The
> predictor is a *guesser* who fills the blanks. The cluster head is a
> *translator* who turns a frame into a vote over `K` boxes.

After training we **keep only the encoder**. The predictor and cluster head were
just training helpers. We throw them away, like scaffolding after a house is
built.

---

## 4. Masking: hiding frames on purpose

Before the encoder runs, we choose some frames to hide. We hide them in
**blocks** (small runs of neighbor frames), until about 65% of the frames are
hidden. The encoder sees zeros at the hidden frames.

Then the predictor puts a special learned **mask token** at the hidden places,
adds position information, and tries to fill them in.

> *Naive question: "Why hide so many frames? 65% is a lot!"*
>
> If we hide too few, the model can cheat by copying neighbors. Hiding a lot
> forces it to truly understand speech to fill the gaps.

---

## 5. Soft targets: the heart of S-JEPA

Here is the key idea. When we ask "what is in this hidden frame?", the old
methods (like HuBERT) answer with **one** cluster number, for example "cluster
42". This is a **hard label**.

But speech is fuzzy at the borders. Between two sounds, a frame can be "half
cluster 42, half cluster 87". A hard label must pick one and throw away the
doubt.

S-JEPA uses a **soft label** instead: a full probability over all `K` clusters,
for example `42 -> 0.55, 87 -> 0.40, others -> small`. This keeps the doubt.

> **Box: where do the soft labels come from?**
>
> From a **Gaussian Mixture Model (GMM)**. A GMM is a set of `K` soft blobs in
> feature space. For a frame, it says how much the frame belongs to each blob.
> Those `K` numbers (they sum to 1) are the soft target.

The predictor outputs its own `K` numbers (a softmax). Training pushes the
predictor's numbers to match the GMM's numbers. The distance between two
probability lists is the **KL divergence**. That single distance is the whole
loss.

```
loss = KL( GMM soft target  ||  predictor softmax )   at hidden frames
```

---

## 6. Two phases (one continuous run)

HuBERT clusters twice: first on simple features, then on better features. S-JEPA
keeps this idea but runs it smoothly, in two phases.

### Phase 1: the frozen MFCC GMM

- We compute **MFCC** features (a classic 39-number summary of sound) for many
  frames.
- We fit one GMM with `K = 100` blobs on these features. It is **frozen**: it
  never changes during Phase 1.
- We train the encoder and predictor to match this GMM at hidden frames.

### Phase 2: the online encoder GMM

- Now the GMM works on the **encoder's own features**, which are richer than
  MFCC. We use `K = 500` blobs.
- The GMM is **online**: after each batch it slowly moves to follow the encoder.
  No need to stop and re-cluster the whole corpus.
- A second, slow copy of the encoder (the **EMA encoder**) feeds the GMM. It
  changes slowly so the targets are stable.
- The active encoder **layer** used by the GMM is chosen automatically, by a
  signal called **effective rank** (how rich a layer is).

> *Naive question: "Why a slow copy of the encoder?"*
>
> If the target and the student change at the same speed, training becomes a dog
> chasing its own tail. The slow copy gives a steadier target to learn from.

---

## 7. How one training step works

For each batch of audio:

1. Make a noisy copy of the wave (augmentation) for the encoder input. Keep the
   clean wave for the target.
2. Build the soft target from the clean wave (MFCC GMM in Phase 1, EMA encoder +
   online GMM in Phase 2).
3. Choose the block mask and run the encoder on the noisy wave.
4. The predictor fills the hidden frames.
5. The cluster head turns predictor frames into `K` logits.
6. Compute the KL loss at hidden frames and do one gradient step.
7. (Phase 2) Update the slow EMA encoder and the online GMM a little.

---

## 8. Training tricks you will see in the config

- **Gradient accumulation.** Small GPUs cannot hold a big batch. We add up the
  gradients of several small batches and step once. The effective batch is
  `batch_size x grad_accum`. We also step at the end of an epoch if some
  gradients are still waiting.
- **Gradient clipping.** We cap the gradient size so one bad batch cannot blow up
  training.
- **Learning rate schedule.** The rate rises during a short *warmup*, then falls
  along a *cosine* curve. Gentle start, gentle end.
- **Checkpoints.** At every epoch we save the full state (model, optimizer,
  scheduler, GMM, EMA). We keep only the newest few and can **resume** exactly
  where we stopped.
- **Best model.** We watch one validation metric (for example `kl`, lower is
  better) and save `best.pt` whenever it improves.
- **History plots.** After each epoch we draw train vs validation curves. If the
  validation curve climbs while the train curve falls, the model is overfitting.

---

## 9. What the metrics mean

- **kl**: how far the prediction is from the soft target. Lower is better. This
  is the main score.
- **top1**: how often the top predicted cluster equals the top target cluster.
- **entropy_bits**: how unsure the prediction is, in bits. `1 bit` means a clean
  two-way tie. The paper shows speech has many such ties at sound borders, which
  is exactly what soft labels keep.
- **effective_rank**: how many directions a layer really uses. Used to pick the
  GMM layer in Phase 2.

---

## 10. After training: inference

Training keeps only the encoder. We export it to **ONNX**, a portable format that
runs fast without Python. Then `infersjepa` loads the ONNX file, reads an audio
clip, and prints the frame features. These features are what you feed to a small
task head (speech recognition, emotion, etc.).

> **You made it!** You now know the whole S-JEPA story: frames, masking, soft GMM
> targets, the KL loss, two phases, and the training tricks. Open
> `cpu/configs/train.yaml`, change a few values, and run `trainsjepa -c
> cpu/configs/train.yaml` to see it live.
