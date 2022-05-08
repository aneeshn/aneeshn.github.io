---
layout: post
title: "Bellman Equations"
date: 2022-05-01 15:58:59 -0200
categories:
  - Research
tags:
  - Reinforcement Learning
  - Transfer Learning
  - MDP
  - Action-Value function
  - Reward function
  - State Space
  - Grid World
  - Machine Learning
  - Artificial Learning
author: Aneesh Chandran
permalink: /post/:categories/:year/:month/:day/
---

 To find an optimal policy the following equations need to be solved

$$\pi(s) = arg\max_{a} \{R(s,a) + \gamma \sum_{s'} P(s' | s, a)V(s') \}$$

$$V^*(s) = \max_{a} \{R(s,a) + \gamma \sum_{s'} P(s' | s, \pi(s))V^*(s') \}$$

here $$V*(s)$$ is the optimal state value function. This can be used to derive the optimal action value function as

$$Q^*(s,a) = \gamma \sum_{s'} P(s' | s, a)V^*(s') \}$$

which can be written out by expanding $$V^*(s) $$ as:

$$Q^*(s,a) = \gamma \sum_{s'} P(s' | s, a)V^*(s') \}$$