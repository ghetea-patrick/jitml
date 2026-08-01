# Just-In-Time Machine Learning (JITML): A Self-Evolving Architecture
**Author:** Ghetea Patrick
**Copyright (c) 2026 Ghetea Patrick. All rights reserved.**
## Abstract
Just-In-Time Machine Learning (JITML) abandons Ahead-Of-Time (AOTML) assumptions in favor of runtime adaptability. By utilizing a pipeline of meta-networks (MNNs), JITML dynamically generates task-networks (TNNs) to handle unfamiliar context on the fly.
## The Limitations of AOTML
Relying primarily on AOTML, current paradigms compute network parameters against finite datasets. When confronted with unfamiliar contexts, AOTML fails to adapt without complete retraining, resulting in brittle deployments in highly dynamic contexts. Furthermore, its lack of structural adaptability forces AOTML models to be either overly generalistic or confined to a single domain of expertise. In contrast, JITML utilizes a baseline configuration that adapts across diverse contexts without requiring retraining, making deployment significantly more practical.
## Core Components of JITML
The JITML paradigm separates task-networks (TNNs) and meta-networks (MNNs), both leveraging feed-forward (FFNN) designs for efficient runtime adaptation. TNNs operate as compact, specialized networks tailored for processing groups of similar contexts, while MNNs are primarily responsible for dynamically generating new TNNs on the fly.
### Task Generation Pipeline (TGP)
When an unfamiliar context is given, it passes through a pipeline of MNNs, where each network governs a single aspect of the new TNN:
 1. **Shape-network (SNN):** Takes the current context as input. Using a sigmoid activation to output a series of Boolean values, it describes the shape of the feed-forward neural network (FFNN), and slots without neurons get purged.
 2. **Activation-network (ANN):** Takes an object describing the shape of the new FFNN, and based on it, chooses the activation functions of each layer.
 3. **Weight-Network (WNN):** Similar to the ANN, it takes the object now populated with activations. This network's output contains the exact values which should be used as weights, and very low-value weights get purged.
As a result, the output neural network is already pruned, making it more compact.
To achieve evolutionary improvements beyond the probabilistic outputs of the MNNs, the TGP relies on a final deterministic section:
 1. **Campaign:** The outputted TNN gets its parameters mutated to generate a given number of candidates (using the analogy of an electoral campaign).
 2. **Voting:** Each candidate gets voted through a set of criteria; the criteria in this case prioritize the cleanliness of the output (having predominant output features or values with a lower entropic score).
 3. **In Office:** The winning output is used by the JITML model, the TNN that generated it is added to a queue of 'tasks' which can be executed, and the current context becomes part of its 'field of expertise.'
```text
╭─────────╮   ╭─────╮   ╭─────╮   ╭─────╮   ╭─────╮
│ Context │──>│ SNN │──>│ ANN │──>│ WNN │──>│ BNN │
╰─────────╯   ╰─────╯   ╰─────╯   ╰─────╯   ╰─────╯
╭──────────╮   ╭────────╮   ╭───────────╮
│ Campaign │──>│ Voting │──>│ In Office │
╰──────────╯   ╰────────╯   ╰───────────╯
╭────────────────────╮
│ TASK NETWORK (TNN) │
╰────────────────────╯

```
## TNN Lifecycle
The lifecycle of a task-network (TNN) within JITML operates on continuous feedback. When any context is inputted, the JITML model will either pick an existing TNN or run the Task Generation Pipeline (TGP). Each time a TNN gets chosen, it becomes 'hotter'; symmetrically, an unused TNN becomes 'colder' and 'colder' until it gets purged to free up resources.
```text
╭─────────╮   ╭────────────────────────────────────────────╮
│ Context │──>│ Does a TNN have this 'field of expertise'? │
╰─────────╯   ╰────────────────────────────────────────────╯
(Yes) ╭──────────────╮
─────>│ Use that TNN │
      ╰──────────────╯
(No)  ╭─────╮
─────>│ TGP │
      ╰─────╯
╭───────────────────────╮   ╭───────────────────────╮   ╭───────────────────╮
│ That TNN gets 'hotter │──>│ The rest get 'colder' │──>│ Purge unused TNNs │
╰───────────────────────╯   ╰───────────────────────╯   ╰───────────────────╯

```
## Meta Training Pipeline (MTP)
Because JITML must handle unknown contexts, we rely on the principle of bootstrapping to train the MNNs. The process uses all known contexts from the Context Bank alongside their respective TNNs:
 1. **Shape-network (SNN):** Trained on this dataset to improve its overall capacity for developing new FFNN architectures.
 2. **Activation-network (ANN):** Separately trained to associate the different outputs of the SNN (the structural object referenced in the TGP) to enforce optimal selections.
 3. **Weight-Network (WNN) & Binary-Network (BNN):** Follow the same training principle, resulting in more refined weight and bias selections.
Once the new MNN suite is ready, it is tested against current pairs of TNNs and contexts to ensure performance has not degraded. As an additional safety measure, it is passed through the same criteria outlined in the TGP.
```text
╭────────────────────────╮   ╭───────────╮   ╭───────────╮   ╭───────────╮   ╭───────────╮
│ Context & Network Pair │──>│ Train SNN │──>│ Train ANN │──>│ Train WNN │──>│ Train BNN │
╰────────────────────────╵   ╰───────────╯   ╰───────────╯   ╰───────────╯   ╰───────────╯
╭────────────────╮   ╭────────────────────────────────╮
│ Check criteria │──>│ Benchmark against current MNNs │
╰────────────────╯   ╰────────────────────────────────╯
╭───────────╮
│ MNN Suite │
╰───────────╯

```
