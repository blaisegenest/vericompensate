# Reviewer 1 (BZVB)

First, we would like to thank you for your constructive review.

## Critical Weaknesses:

> Weakness 1. The certified guarantee is about a different model than the deployed task requires.

As emphasized throughout the paper, we propose to *deploy* the AE-DNN rather than the original vanilla DNN (e.g., in the abstract: "We propose alternative 'AE-DNNs' to vanilla DNNs to use in exploitation"). 
As you rightly notes, *"The paper shows global bounds on vanilla DNNs are irreducibly pessimistic."* Thus, to achieve real-time certification, the original vanilla DNN cannot be used during deployment.

> Weakness 2. No adaptive attacks against the projected model. Robustness claims on non-ID regions rest only on indirect evidence (bounds, visualizations). No attack success rate is reported for LAE-DNNs. For a paper whose central contribution is an architecture filtering non-ID inputs, adversarial validation of exactly that boundary is the minimum expected evidence.

Our central claim is not that the AE-DNN is inherently more robust on in-distribution data, but that it enables tighter bounds (which it does). Still we agree with you that the robustness numbers are interesting. 

We provide two tests: 
1. original dataset: only in-distribution data, for which we do not expect much change? [EC change on what? on accuracy?]
2. random images: mostly out-of-distribution data, where we expect AE-DNN to be more robust by filtering perturbations. 

These results are summarized in the table below and have been incorporated into the revised manuscript. ___TODO TODO__

> Weakness 3. Offline cost is unbounded and unreported. The tightest LAE-CIFAR10 bound required a 60-hour Gurobi timeout; total cost over all 45 class pairs, and re-computation after retraining, are never quantified. Combined with small networks and linear-AE-only experiments, feasibility beyond toy scale remains an assumption, not a finding.

Constructing the AE-DNN offline takes negligible computation time ($X$ seconds). 

As per Table 3, precomputing the bounds across all 45 classes [_EC_ 10 ? CIFAR10-Conv?]  (2 runs) requires 120 hours (60h or 14400s per run) using 23 parallel threads on a 32-core CPU for the longest timeout setting (this holds for both Diff MILP and ITNE).

Using additional CPU cores or machines would directly reduce this runtime (e.g., 60 hours). Importantly, this is a *one-time cost*, since the network is fixed for deployment. 

Also, note that extending the solver timeout from 4 hours (14400s) to 60 hours does not significantly improve the bounds. The impact on the the real-time certification rate would be only a few percentage points: the computational cost can be reduced without much affecting the result. What we suggest is thus to use a feasible runtime and proceed with the bounds produced. 

## Answers to Questions For Authors:

> In what precise sense is the resulting guarantee a robustness certificate "for the model in deployment"?

The model in deployment is the AE-DNN itself, hence maintaining high accuracy on the AE-DNN is crucial.

> Please state the certified perturbation set explicitly,

The certified perturbation set is any continuous image bounded by a given norm ($L_\infty$, $L_1$, or a conjunction of both), with each pixel values constrained to $[0,1]$.

> whether a component-wise adversary exploiting exactly these filtered directions is considered in-scope, out-of-scope, or excluded by assumption.

An adversary can exploit any direction in the input space, so in particular it can use filtered directions. As you remark, the first 2 layers (encode + decode) of AE-DNN would project away these components. 
So, for the AE-DNN this makes no difference, but it can vary it.
[[EC: Exploiting these filtered directions is out-of-scope of the present work, yet the bounds remain significantly tighter with our method.]]

> The paper's central claim concerns behavior on non-ID inputs, yet no attack success rate against LAE-DNNs is reported. Did you run any adaptive attack (e.g., PGD or a targeted attack through the projection P, or an attack searching in latent space) on the deployed LAE-DNNs?

We evaluated our models using $\alpha,\beta$-CROWN, which has its own attack, and, if not sucessful, tries to prove robustness. The proportion of unknown cases is very low: the attack itself would not change the results.

> Accuracy loss reaches ≤1% only at latent dimension 60 for CIFAR-10, after a plateau of 5–16% at smaller dimensions. Do you have any evidence (even preliminary) on whether the ≤1% budget remains attainable at ImageNet-scale inputs, or does the required latent
dimension grow superlinearly with data complexity?

We did not yet experiment on DNNs for ImageNet. [[EC: Extending our framework for higher-dimensional datasets is a key direction for future work.]]

# Review 2 (MLsQ)

First, we would like to thank you for your constructive review.

>The main certification improvements rely on prepending an LAE,
>effectively changing the network being verified.

First, as we show in the paper, the original DNN is not amenable to
real-time verification (c.f. the review by reviewer BZBD).

That is the reason why the DNN we use *in exploitation* is the AE-DNN
(which is only a slight variation of the original DNN).
It has similar accuracy as hte original DNN, and it is amenable to
real-time verification, unlike the original DNN.

>The improved certification guarantees apply only to the modified
>networks, not the original DNNs.

Of course, since it is impossible to have such guarantee on the
original DNN as we show in the paper. This is the reason why we use the
AE-DNN in deployment (see e.g. the abstract " for AE-DNNs
used in exploitation").

> Similar classification accuracy does not imply
equivalent predictions

For networks with very high accuracy, it does (there is only one switch
between correct and incorrect).

> or robustness properties.

Notice that what we are interested in is certified real-time
robustness, not plain robustness.
Still, we completed the pdf and report the different
robustness level.

__TODO TODO__

> Furthermore, projecting inputs onto a low-dimensional subspace does
> not necessarily eliminate out-of-distribution behavior, weakening the motivation for
> LAE-DNNs.

The motivation is to obtain smaller bound. Which is a clear success in
all DNNs we considered, see Table XX.

Eliminating out-of-distributioin behavior is not the goal (and indeed,
it can never be fully reach), it is the rationale to obtain the goal
(obtain smaller bounds), which is fully successful: bouds are 3-5
times smaller on the LAE-DNN than on the original DNN.


> MILP scales poorly to large DNNs due to the combinatorial complexity
> of unstable ReLUs, and the Diff formulation inherits this limitation.
> LP relaxation avoids the combinatorial search but may yield overly loose bounds, while large LPs remain computationally expensive. Consequently, the global bounds may be too loose for effective online certification, regardless of its low overhead. Scalable abstract interpretation techniques for computing global differential bounds, such as DiffPoly (PLDI 2024), might be a more promising direction.

We agree that for larger DNNs, other (faster bu not as
accurate) techniques  might be useful. This is out of scope of the
current paper.

Concerning DiffPoly specifically, the setting is different: they
consider 2 different DNNs on the same input, while we consider the same
DNN on 2 slightly different inputs. It is non trivial to change DiffPoly
to tackle our case, reason why we did not report experiments.
Still, we run some experiment to compare their speed and accuracy on
DiffPoly Problem. We considered 3 different complexity problems to
compare our tool with DiffPoly:

X
Y
Z

As conclusion, we can see that Diff is more accurate but slower
than DiffPoly.


> The real-time certification experiments (Table 4) only consider L_1 perturbations, while the L_inf experiments (Table 1) report global bounds. The chosen L_1 budgets exclude many dense perturbations permitted under standard L_inf settings, which raises concerns about the generalizability of the reported results.

L_1 and L_infty are orthogonal norms. We believe a L_1 perturbation of 1 is meaningful (it means one pixel can be fully changed freely, with proven attacks adding patch
anywhere on the image). We can provide real-time certification up to
L_1 perturbation of 4 (4 full pixels can be change fully, or 8 pixel
can be altered considerably, etc). Such alteration cannot be modelled by
Linfty perturbation. L_infty perturbation are computationaly easier to handle than L_infty perturbation (upper and lower bounds are much tighter in Table 1
than in Table 3).

__Liao, can you try on LAE-FMNIST-Conv to run Linfty with epsilon = 0.01? And see what it gives__


## Answer to Questions:

>Q1: What practical scenarios justify replacing existing DNNs with LAE-DNNs, given that the robustness guarantees do not transfer to the original networks?

DNNs should not be replaced by LAE-DNN only in runnign the bounds. It has to be replaced in exploitation / deployement also, which is exactly what we propose explicitly in the paper (see. e.g. the abstract).

>Q2: How is the global input domain defined and encoded in MILP to
ensure sound certification?

e.g. for MNIST: 784 continuous variables (one for each pixel), each
pixel can take any value in [0,1]. This is the "full" domain, input of
both LAE-DNN and original DNN.

# Review 3 (SpAJ)

We would like to thank you for the constructive review.

## Critical Weaknesses:

>Table 3 shows that introducing LAE can substantially increase the
relative gap between the upper and lower bounds, as the lower bounds
decrease much more dramatically than the upper bounds. In particular,
for LAE-CIFAR10-Conv, the lower bound is 0.0015, whereas the upper
bound remains 0.99 even after 60 hours of computation. The reason for
this extremely large gap is not adequately explained.

LAE-DNN need additional veriables to encode the LAE part compared with
the original DNN. Althgough these variables are all continuous, so not
too costly by themselves, they still make the overall problem
more complicated. Judging by the results, we strongly believe that the
lower bounds are far away from the real bounds. On the other hand, the
upper bound seem close to a realistically boundary.

As what is crucial are the upper bounds, and lower bounds are
only indicative, this is not a major point we worked on.

> The paper does not sufficiently explain why PCA-based dimensionality
> reduction improves the robustness bounds. In particular, it remains
> unclear 
whether the improvement is specific to the subspace selected by PCA or
simply results from dimensionality reduction itself.

Dimensionality itself does not explain why the bounds are better.
Any space allowing the example in figure X left (dimension=2 suffices)
would result in the very high upper bound reported.

We explain our rationale: the dimensions are computed from the in
distribution samples, so that it represent in distribution, and out of
distribution dimensions are projected away. Hence out of distribution,
on which the DNN behaves badly cannot affect the result of LAE-DNN
anymore.

> Although Table 4
reports certification rates, it does not provide certified accuracy or
classification accuracy on certified inputs. Thus, it is difficult to
evaluate the relationship between classification correctness and
robustness certification.

We report these number in Table XX in the revised PDF.


> Questions For Authors: What causes the
extremely large gap between the lower and upper bounds for
LAE-CIFAR10-Conv in Table 3? Is this due to a genuinely small
worst-case sensitivity, or to difficulty finding good feasible
solutions or obtaining tight relaxations in the MILP solver?

From our experiments, it is very hard to find feasible solution for
LAE-CIFAR10-Conv.