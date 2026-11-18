# Official Review of Submission31690 by Reviewer BZVB
Official Reviewby Reviewer BZVB25 Oct 2026, 22:36 (modified: 06 Nov 2026, 20:28)EveryoneRevisions

## Paper Summary:

This paper proposes a two-step method for real-time certification of global DNN robustness. First, global bounds on output differences between class pairs are computed offline with an MILP model. Second, each incoming image is certified online by comparing class output gaps against these precomputed bounds. The check needs about 27 CPU instructions, roughly 0.5 ms. The key obstacle is that global bounds on vanilla DNNs are overly pessimistic, because non-distribution inputs cause erratic behavior. The authors show no verification method can fix this problem on unmodified DNNs.

The paper makes two contributions. First, the AE-DNN architecture. A linear autoencoder, trained on in-distribution data, projects inputs onto a small latent subspace before the DNN. This removes non-ID directions that inflate global bounds. Bounds shrink 3–15x with at most 1% accuracy loss. Appendix C proves the projection can be absorbed into the first DNN layer, so no architecture change is needed. Second, the Diff MILP model. It represents input-perturbation differences as explicit variables, giving tighter global bounds than classical encodings.

Experiments cover MNIST, Fashion-MNIST, and CIFAR-10 with four pretrained DNNs and their LAE variants. Bounds are computed with Gurobi on a 32-core Threadripper, with timeouts up to 216000 seconds. LAE-DNNs reach bounds 15x, 3x, and 7x smaller than their vanilla counterparts. Notably, the LAE-CIFAR10 bound (0.99) is below the vanilla CIFAR10 lower bound (3.38), so no method can match LAE bounds on the original network. Real-time certification exceeds 70% of images for L1 perturbations of 1, 4, and 2, at 0.5 ms latency.

Presentation: 3: Clear and easy to follow.
Informativeness: 2: I learned something interesting or useful.
Soundness: 3: I have no substantive concerns about the paper’s soundness.

## Critical Strengths:

1. A well-posed, provably reframed problem. The paper shows global bounds on vanilla DNNs are irreducibly pessimistic due to non-ID inputs, then proves (Appendix C) a rank-n input projection can be absorbed into the first layer. The tightness gain is architectural, not algorithmic—no verification method could achieve it on the unmodified network.

2. A decisive empirical argument. LAE-CIFAR10's upper bound (0.99) falls below vanilla CIFAR10's lower bound (3.38). This converts a method comparison into an impossibility argument for competing tools on the original network, which is rare and convincing.

3. Real-time practicality with good scoping. The expensive work is fully offline; the online check costs ~27 CPU instructions (0.5 ms, one core), certifying 70%+ of images at up to 4x larger radii, for ≤1% accuracy loss. Authors self-declare concept-and-feasibility scope, keeping claims properly bounded.

## Critical Weaknesses:

1. The certified guarantee is about a different model than the deployed task requires. Bounds are computed on the projected network. Perturbation components orthogonal to the latent subspace are invisible to the network—filtered out, not resisted. The certified perturbation set is silently redefined. The paper never states this trade explicitly.

2. No adaptive attacks against the projected model. Robustness claims on non-ID regions rest only on indirect evidence (bounds, visualizations). No attack success rate is reported for LAE-DNNs. For a paper whose central contribution is an architecture filtering non-ID inputs, adversarial validation of exactly that boundary is the minimum expected evidence.

3. Offline cost is unbounded and unreported. The tightest LAE-CIFAR10 bound required a 60-hour Gurobi timeout; total cost over all 45 class pairs, and re-computation after retraining, are never quantified. Combined with small networks and linear-AE-only experiments, feasibility beyond toy scale remains an assumption, not a finding.

## Questions For Authors:

The bound β̄ is computed on the projected network, so perturbations orthogonal to the rank-n latent subspace leave the output unchanged. In what precise sense is the resulting guarantee a robustness certificate "for the model in deployment"? Please state the certified perturbation set explicitly, and clarify whether a component-wise adversary exploiting exactly these filtered directions is considered in-scope, out-of-scope, or excluded by assumption.

The paper's central claim concerns behavior on non-ID inputs, yet no attack success rate against LAE-DNNs is reported. Did you run any adaptive attack (e.g., PGD or a targeted attack through the projection P, or an attack searching in latent space) on the deployed LAE-DNNs?

Accuracy loss reaches ≤1% only at latent dimension 60 for CIFAR-10, after a plateau of 5–16% at smaller dimensions. Do you have any evidence (even preliminary) on whether the ≤1% budget remains attainable at ImageNet-scale inputs, or does the required latent dimension grow superlinearly with data complexity?
Flag For Ethics Review: No
Rating: 3: Weak acceptance
Confidence: 3: I am familiar with this topic and its relevant literature.
Add:

# Official Review of Submission31690 by Reviewer MLsQ
Official Reviewby Reviewer MLsQ20 Oct 2026, 22:10 (modified: 06 Nov 2026, 20:28)EveryoneRevisions

## Paper Summary:

The paper proposes real-time DNN robustness certification using global robustness bounds computed offline via a Diff MILP formulation. During inference, these bounds are used to certify robustness through simple output comparisons. To obtain tighter bounds, the authors prepend a linear autoencoder (LAE) to the original DNN. Experiments on MNIST, Fashion-MNIST, and CIFAR-10 show improved global bounds and high certification rates for the modified networks.
Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 2: I learned something interesting or useful.
Soundness: 3: I have no substantive concerns about the paper’s soundness.

## Critical Strengths:

    Leveraging precomputed global robustness bounds for efficient robustness certification during inference is interesting and potentially useful.

## Critical Weaknesses:

    The main certification improvements rely on prepending an LAE, effectively changing the network being verified. The improved certification guarantees apply only to the modified networks, not the original DNNs. Similar classification accuracy does not imply equivalent predictions or robustness properties. Furthermore, projecting inputs onto a low-dimensional subspace does not necessarily eliminate out-of-distribution behavior, weakening the motivation for LAE-DNNs.

    MILP scales poorly to large DNNs due to the combinatorial complexity of unstable ReLUs, and the Diff formulation inherits this limitation. LP relaxation avoids the combinatorial search but may yield overly loose bounds, while large LPs remain computationally expensive. Consequently, the global bounds may be too loose for effective online certification, regardless of its low overhead. Scalable abstract interpretation techniques for computing global differential bounds, such as DiffPoly (PLDI 2024), might be a more promising direction.

    The real-time certification experiments (Table 4) only consider L_1 perturbations, while the L_inf experiments (Table 1) report global bounds. The chosen L_1 budgets exclude many dense perturbations permitted under standard L_inf settings, which raises concerns about the generalizability of the reported results.

## Questions For Authors:

    Q1: What practical scenarios justify replacing existing DNNs with LAE-DNNs, given that the robustness guarantees do not transfer to the original networks?

    Q2: How is the global input domain defined and encoded in MILP to ensure sound certification?

Flag For Ethics Review: No
Rating: 1: Clear rejection
Confidence: 4: I have substantial direct research experience on this topic.
Add:

# Official Review of Submission31690 by Reviewer SpAJ
Official Reviewby Reviewer SpAJ18 Oct 2026, 14:51 (modified: 06 Nov 2026, 20:28)EveryoneRevisions

## Paper Summary:

This paper proposes a method for real-time certification of neural network robustness by computing global robustness bounds offline and using them for efficient online certification. While the basic framework of dividing certification into offline and online stages has already been proposed in prior work, such as VHAGaR, the authors introduce two techniques to improve its practicality. The first is Diff MILP, which directly represents differences in neuron activations to compute tighter upper bounds. The authors also develop two computationally efficient abstractions, 2b+1 and 1b, of Diff MILP. The second is the incorporation of a PCA-based Linear AutoEncoder (LAE) into neural networks to reduce global robustness bounds. The effectiveness of these methods is supported by numerical experiments.
Presentation: 3: Clear and easy to follow.
Informativeness: 2: I learned something interesting or useful.
Soundness: 2: I have minor or fixable concerns about the paper’s soundness.

## Critical Strengths:

    Diff MILP achieves tighter upper bounds than conventional MILP encodings. The proposed 2b+1 and 1b models also provide useful trade-offs between computational efficiency and bound tightness.
    LAE substantially reduces the global robustness bounds.

## Critical Weaknesses:

    Table 3 shows that introducing LAE can substantially increase the relative gap between the upper and lower bounds, as the lower bounds decrease much more dramatically than the upper bounds. In particular, for LAE-CIFAR10-Conv, the lower bound is 0.0015, whereas the upper bound remains 0.99 even after 60 hours of computation. The reason for this extremely large gap is not adequately explained.
    The paper does not sufficiently explain why PCA-based dimensionality reduction improves the robustness bounds. In particular, it remains unclear whether the improvement is specific to the subspace selected by PCA or simply results from dimensionality reduction itself.
    Although Table 4 reports certification rates, it does not provide certified accuracy or classification accuracy on certified inputs. Thus, it is difficult to evaluate the relationship between classification correctness and robustness certification.

## Questions For Authors:

    What causes the extremely large gap between the lower and upper bounds for LAE-CIFAR10-Conv in Table 3? Is this due to a genuinely small worst-case sensitivity, or to difficulty finding good feasible solutions or obtaining tight relaxations in the MILP solver?
    Is the reduction in robustness bounds due to properties of the subspace selected by PCA, rather than dimensionality reduction alone? Could the authors investigate this by comparing PCA with random projections at comparable classification accuracy?
    Could the authors report certified accuracy in Table 4, as well as classification accuracy separately for certified and uncertified inputs, to better evaluate the practical significance of robustness certification?

Flag For Ethics Review: No
Rating: 3: Weak acceptance
Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.