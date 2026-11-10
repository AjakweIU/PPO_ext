# ICTC 2026 Reviewer Comments

This file is a UTF-8 Markdown conversion of the original reviewer-comments document for direct use in VS Code and AI coding assistants.

## Reviewer 1

The paper compares the performance of the PPO agent against only a single baseline: a 'fixed rule-based heuristic.'

To truly demonstrate the practical superiority of reinforcement learning, it would be highly beneficial to validate it through comparisons with optimization algorithms widely used in the industry, such as Mixed Integer Linear Programming (MILP) or Model Predictive Control (MPC).

Furthermore, if the system is expanded to a large-scale, national-level CCUS pipeline network, the action space will explode exponentially. Therefore, exploring strategies to ensure the necessary scalability to handle this complexity would be an excellent direction for future research.

## Reviewer 2

1.	The simulation environment needs to be described in more detail. The paper states that source availability, sink demand, route capacity, energy conditions, and leakage events are synthetically generated, but their distributions, temporal dynamics, and constraint ranges are not provided. Since the PPO results depend strongly on the environment design, these details are necessary to understand and reproduce the reported gains.

2.	The reward function is only described at a high level. The values of the weights (w_q), (w_e), (w_v), and (w_l) are not reported, and the normalization of the throughput and energy terms is unclear. The authors should provide these values and explain how they were selected. Otherwise, it is difficult to determine whether the large reward improvement mainly reflects the chosen reward scaling.

3.	The PPO agent is compared with only one fixed heuristic baseline. Since the reported throughput improvement is as high as 89.9%, the heuristic may be too weak. At least one additional baseline, such as a greedy scheduler, a mathematical optimization method, or another RL algorithm, would make the evaluation more convincing.

4.	It is unclear whether the normal, high-demand, and energy-spike evaluation scenarios differ from the conditions encountered during PPO training. If the training environment already includes the same scenario patterns, the results mainly show interpolation rather than generalization. The authors should clarify the train/evaluation scenario separation.

5.	Fig. 4 presents throughput as increasing over the PPO training episodes. Please clarify whether each point is the evaluation throughput of the current policy or the throughput observed during training. The comparison with the heuristic should also be shown directly for the throughput metric, not only for the reward.

## Reviewer 3

This paper investigates a PPO-based reinforcement learning framework to operational decision-making in a CCUS pipeline network. The topic is relevant, and the reported improvements indicate the potential of adaptive learning-based control. However, the current experimental design and methodological descriptions are insufficient to establish the practical superiority and generalizability of the proposed approach.

The following issues should be addressed.

1. The simulation environment requires a more comprehensive description.

2. The reward-function design should be presented more transparently.

3. The PPO agent is evaluated against only a fixed rule-based heuristic, which may be too weak to support the claimed performance superiority, particularly given the reported throughput gain of 89.9%. The evaluation should include stronger benchmarks, such as a greedy scheduler, MILP, MPC, or another representative reinforcement learning algorithm. Comparisons of computational complexity and decision-making time would further clarify the practical advantages of PPO.

To work on the following aspects of the review comments.

- To review tables and figures in the manuscript after taxonomy recoding and evidence synthesis

- The work on the threat model lifecycle × architecture × property × vector × mitigation × residual-risk matrix as requested by the reviewer

---

## Implementation Scope Note

For the AI-only journal extension, use only reviewer comments relevant to:

- simulation-environment detail and reproducibility;
- reward-function transparency and normalization;
- stronger baselines such as greedy scheduling, MILP/MPC, or another RL method;
- explicit train/validation/test separation and generalization;
- direct throughput comparison against baselines;
- scalability of the action/decision space;
- computational complexity and decision-making time.

The final two notes in the original document concerning taxonomy recoding and a threat-model matrix appear unrelated to this CO₂/PPO manuscript and should not be treated as AI-journal implementation requirements unless separately confirmed.
