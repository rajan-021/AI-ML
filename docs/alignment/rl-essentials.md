---
title: RL Essentials
description: Just enough reinforcement learning for RLHF and reasoning models - MDPs, value functions and Bellman equations, Q-learning and DQN, policy gradients, actor-critic, PPO, and bandits.
act: 4
priority: Medium
tags:
  - Applied Scientist
  - ML Engineer
  - AI Engineer
---

# RL Essentials

!!! abstract "TL;DR"
    In reinforcement learning an agent acts, receives rewards, and learns a **policy** that maximizes expected discounted return. **Value-based** methods learn how good states and actions are (Bellman equations, Q-learning, DQN). **Policy-gradient** methods adjust the policy directly to make high-reward actions more likely; a **critic** reduces their variance; **PPO** clips each update to keep training stable. That's the machinery of RLHF: the LLM is the policy, tokens are actions, and a reward model scores the response. **Bandits** are one-step RL used in ads and recommendations.

!!! evolution "Where this fits"
    **Before:** supervised learning needs labels; some problems only give delayed rewards. **This page:** studied here because [Post-Training & Alignment](post-training.md) and [Reasoning Models](reasoning.md) are built directly on it.

## The setup

An agent observes state $s_t$, takes action $a_t$ from its **policy** $\pi(a \mid s)$, receives reward $r_{t+1}$ and moves to $s_{t+1}$. Formally a **Markov decision process** $(S, A, P, R, \gamma)$: the next state depends only on the current state and action.

| Term | Meaning |
|---|---|
| **Return** | $G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$, discounted future reward; $\gamma \in [0, 1)$ trades near-term against long-term reward |
| **State value** | $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ |
| **Action value** | $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$ |
| **Advantage** | $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$: how much better this action is than average |
| **Exploration vs exploitation** | Try new actions to learn, or use the best known one |
| **On-policy vs off-policy** | Learn about the policy generating the data, or about a different (usually greedy) one |

**Why RL is hard:** rewards are delayed (credit assignment), data depends on the policy (non-stationary), exploration is needed, and training is high-variance and sample-hungry.

## Value functions and Bellman equations

Values satisfy a recursive relationship: the value now equals the immediate reward plus the discounted value of what follows.

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi,\, s'}\left[r + \gamma V^\pi(s')\right]
\qquad
Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a')\right]
$$

The second (Bellman optimality) equation defines the optimal action values; acting greedily on $Q^*$ is optimal.

| Method | Uses a model of the environment? | Updates from | Bias / variance |
|---|---|---|---|
| **Dynamic programming** (value / policy iteration) | Yes | Full expectations | Exact, but needs the model |
| **Monte Carlo** | No | Complete episode returns | Unbiased, high variance |
| **Temporal difference (TD)** | No | One step plus a bootstrapped estimate: $r + \gamma V(s')$ | Biased, lower variance; learns online |

## Q-learning and DQN

**Q-learning** (Watkins, 1989) is off-policy TD control:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
$$

It learns the optimal Q-values while behaving with an exploratory policy such as **ε-greedy** (random action with probability ε). **SARSA**, the on-policy cousin, uses the action actually taken next, $r + \gamma Q(s', a')$, and learns more cautious policies when exploration is risky.

**DQN** (2013–2015) uses a neural network for $Q$ and played Atari games from pixels. Two tricks made it stable:

- **Experience replay:** store transitions and train on random minibatches, breaking correlations between consecutive samples and reusing data.
- **Target network:** compute targets with a periodically updated copy of the network, so the target doesn't chase itself.
- **Double DQN** reduces the max operator's overestimation bias by choosing actions with one network and evaluating them with the other.

**Limits:** needs a max over actions, so it suits discrete action spaces; generating text (a huge, sequential action space) calls for policy methods.

## Policy gradients

**The problem:** value-based methods struggle with large or continuous action spaces and with stochastic policies.

**Core idea:** parameterize the policy $\pi_\theta$ and climb the gradient of expected return. The **log-derivative trick** gives:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\right]
$$

**REINFORCE** (Williams, 1992) estimates this with sampled episodes: increase the log-probability of actions in proportion to the return that followed. It's unbiased but very noisy.

**Variance reduction:**

- **Baselines:** subtract a baseline $b(s)$ from the return; it doesn't change the expected gradient (because $\mathbb{E}[\nabla \log \pi]\,b = 0$) but reduces variance. The natural baseline is $V(s)$, which turns $G_t - V(s_t)$ into an estimate of the **advantage**.
- **Actor–critic:** an actor (the policy) and a critic (a learned value function) trained together; the critic supplies advantages.
- **GAE** (generalized advantage estimation) blends multi-step TD errors $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ as $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$, trading bias against variance with $\lambda$.
- **Entropy bonus:** keeps the policy from collapsing too early, preserving exploration.

## PPO

**The problem:** a policy-gradient step that's too large can wreck the policy, and the bad policy then collects bad data. TRPO constrained each step with a KL trust region but is complex.

**Core idea:** optimize a surrogate objective that **clips** the probability ratio, so the new policy can't move too far from the one that collected the data:

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t(\theta)\,\hat A_t,\ \text{clip}\left(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\right)\hat A_t\right)\right],
\qquad \rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

- With positive advantage, the objective stops rewarding increases in probability beyond $1 + \epsilon$; with negative advantage, beyond $1 - \epsilon$. Typical $\epsilon$ is 0.1–0.2.
- Each batch of experience is reused for several epochs of minibatch updates, improving sample efficiency.
- The full loss adds a value-function loss for the critic and an entropy bonus. Some variants use an adaptive KL penalty instead of clipping.
- Simple, robust and general, PPO became the default RL algorithm, including for RLHF.

## Beyond: model-based and offline RL

- **Model-based RL** learns or uses a model of the environment to plan. **AlphaGo** and **AlphaZero** combine Monte Carlo tree search with policy and value networks trained by self-play; MuZero also learns the model.
- **Offline RL** learns from a fixed logged dataset without new interaction, which suits domains where exploration is costly (healthcare, logged recommendations), but must avoid trusting actions the data never tried.

## Bandits

A **multi-armed bandit** is RL with a single step: choose an arm, get a reward, no state transitions. The goal is to minimize **regret** (reward lost versus always choosing the best arm).

| Strategy | Idea |
|---|---|
| **ε-greedy** | Exploit the best arm, explore at random with probability ε |
| **UCB** | Choose the arm with the highest optimistic estimate: mean plus $\sqrt{2\ln t / n_a}$, so uncertain arms get tried |
| **Thompson sampling** | Keep a posterior per arm (for click rates, a Beta distribution), sample from each, pick the highest sample |
| **Contextual bandits** (LinUCB, neural bandits) | Condition on user and item features; standard for personalization and ads |

Bandits are used for headline and creative testing, exploration slots in recommenders, and adaptive experiments.

## RL for language models

| RL concept | In RLHF / RL for LLMs |
|---|---|
| Policy | The LLM, $\pi_\theta(\text{token} \mid \text{prompt, tokens so far})$ |
| State | Prompt plus tokens generated so far |
| Action | The next token |
| Episode | Generating one complete response |
| Reward | A reward model's score (or a verifier's pass/fail) at the end, minus a per-token KL penalty against a reference model |
| Critic | A value model estimating expected reward from partial responses (PPO); GRPO replaces it with group statistics |

Details are on [Post-Training & Alignment](post-training.md) and [Reasoning Models](reasoning.md).

## Interview questions

??? question "Explain the Bellman equation."
    It expresses a value recursively: the value of a state (or state–action pair) equals the expected immediate reward plus the discounted value of the next state. For a policy, $V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s')]$. The optimality version, $Q^*(s, a) = \mathbb{E}[r + \gamma \max_{a'}Q^*(s', a')]$, defines the best achievable values. Dynamic programming solves these equations with a known model; TD methods and Q-learning approximate them from samples.

??? question "On-policy vs off-policy learning."
    On-policy methods learn the value or improve the policy that is generating the data (SARSA, REINFORCE, PPO): stable, but data from older policies must be discarded or corrected. Off-policy methods learn about a target policy (often the greedy one) from data generated by a different behavior policy (Q-learning, DQN with replay): more sample-efficient because old data can be reused, but can be less stable and needs corrections such as importance sampling for policy methods.

??? question "Monte Carlo vs temporal-difference learning."
    Monte Carlo waits for the episode to end and uses the actual return as the target: unbiased but high variance, and it needs episodes to finish. TD updates after each step with a bootstrapped target $r + \gamma V(s')$: biased because it relies on the current estimate, but much lower variance, works online and in continuing tasks. TD(λ) and n-step returns interpolate between them.

??? question "Why does DQN use experience replay and a target network?"
    Consecutive transitions are highly correlated and the data distribution shifts as the policy changes, which destabilizes neural network training; sampling random minibatches from a replay buffer breaks correlations and reuses data. And because the TD target uses the same network being updated, the target moves with every step and training can oscillate or diverge; a target network updated only periodically holds targets fixed for a while.

??? question "Derive the policy gradient."
    $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int p_\theta(\tau)R(\tau)\,d\tau$. Then $\nabla J = \int \nabla p_\theta(\tau)R(\tau)\,d\tau = \int p_\theta(\tau)\nabla\log p_\theta(\tau)R(\tau)\,d\tau$ using $\nabla p = p\,\nabla\log p$. The trajectory log-probability is the sum of log policy terms plus environment dynamics, which don't depend on $\theta$, so $\nabla J = \mathbb{E}[\sum_t \nabla\log\pi_\theta(a_t \mid s_t)R(\tau)]$. Using the return from time $t$ onward (causality) gives REINFORCE.

??? question "Why subtract a baseline in policy gradients?"
    Raw returns make gradient estimates very noisy: every action in a good episode gets pushed up, even bad ones. Subtracting a baseline that depends only on the state keeps the gradient unbiased, since $\mathbb{E}_{a \sim \pi}[\nabla\log\pi(a \mid s)]\,b(s) = 0$, but greatly reduces variance. Using $V(s)$ as the baseline means actions are reinforced according to their advantage, how much better than expected they turned out.

??? question "What does PPO's clipping do?"
    It limits how far one update can move the policy from the policy that collected the data. The probability ratio between new and old policies is clipped to $[1 - \epsilon, 1 + \epsilon]$, and the objective takes the minimum of the clipped and unclipped terms, so there's no benefit in pushing an action's probability further once the ratio leaves that range. This approximates a trust region cheaply, prevents destructive updates, and allows several epochs of updates on each batch.

??? question "Explain exploration vs exploitation and compare ε-greedy, UCB and Thompson sampling."
    Exploitation picks the best-known option; exploration tries others to learn whether they're better. ε-greedy explores uniformly at random with probability ε: simple but wastes exploration on clearly bad arms. UCB adds an optimism bonus that shrinks as an arm is tried more, so it explores uncertain arms systematically. Thompson sampling samples from each arm's posterior and picks the best sample, exploring in proportion to the probability each arm is optimal; it often performs best in practice and handles delayed feedback well.

??? question "How is text generation framed as RL in RLHF?"
    The LLM is the policy; the state is the prompt plus the tokens generated so far; each action is the next token; an episode is one complete response. A reward model scores the finished response, and a per-token KL penalty against the reference (SFT) model is subtracted so the policy doesn't drift into reward hacking or gibberish. PPO maximizes this reward using a value model as the critic for advantage estimates.

??? question "What is the credit assignment problem?"
    When a reward arrives only at the end (a game won, a response judged good), it's unclear which earlier actions deserve the credit. RL handles it with discounting, value functions that estimate future reward from intermediate states, advantage estimation (GAE), and, for LLMs, process rewards that score intermediate reasoning steps rather than only the final answer.

!!! pitfall "Common mistakes"
    - Mixing up on-policy and off-policy methods (Q-learning is off-policy; PPO is on-policy).
    - Forgetting that a state-dependent baseline doesn't bias the policy gradient.
    - Describing PPO's clip as clipping gradients.
    - Treating each RLHF token as receiving its own reward-model score (usually only the final token does, plus per-token KL).

## Revision checklist

- [ ] Define return, V, Q and advantage
- [ ] Write the Bellman expectation and optimality equations
- [ ] Write the Q-learning update and explain DQN's two tricks
- [ ] Derive the policy gradient and explain baselines and actor–critic
- [ ] Write PPO's clipped objective and explain why it's stable
- [ ] Compare ε-greedy, UCB and Thompson sampling
- [ ] Map RL terms onto RLHF

## Go deeper

- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
- Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (free)
- [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)
- [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602)
