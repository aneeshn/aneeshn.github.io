---
layout: post
title: "A quick intro to Reinforcement Learning and Q-Learning"
date: 2022-04-29 15:58:59 -0200
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


<figure>
<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 40%;"
    src="/assets/gridworld.png" 
    alt="Simple Grid world"/>
<figurecaption style="display: block; 
           margin-left: auto;
           margin-right:auto;
           text-align: center;">Simple Grid World</figurecaption>
</figure>

Picture the simple `3 x 3` grid world above. In this grid world an agent can only move in the cardinal directions: `up, down, left and right`. There is only one goal state the agent needs to reach. Upon reaching this goal state the agent has achieved it's objective. 

For this post I am interested in using Reinforcement Learning (RL) to solve this domain. Lets use the Markov Decision Process (MDP) framework as done in RL. An MDP can be written as a tuple:
```java
M = <S, A, P(s'| s, a), R(s, a)>
```
where 

- a set of States `s ∈ S`
- a set of Actions `a ∈ A`
- an action dependent state transition probablity function `` P(s'| s, a)``
- a reward function `R(s, a) -> ℝ `

If the agent naively moves from one state to another, taking random actions in every state it is in, it will eventually land up in the goal state causing for it to end its "turn". When the agent does this it has generated a sequence of states and actions, this known as a **trajectory**. 

What if we would like to know if there are specific trajectories that are more important than others? In order to do this we need a way to "score" these trajectories. If we sum up all the reward we get from each state until the trajectory ends then that should be good enough right? What if this sequence is really long? We should not assign credit from a state far into the sequence to impact the score of the trajectory. We should probably give more weight to the rewards closer to the beginning to the sequence than further back. I am trying not to go into further detail, I encourage the reader to do some research into why this is the case. So, we define a cumulative reward function as 

<img src="https://latex.codecogs.com/gif.latex?G = \sum^T_{t=0}\gamma^t R(s_t, a_t)"
style="display: block; 
           margin-left: auto;
           margin-right: auto;"/>

In order to solve an MDP `M` we need to find an optimal policy function `π(s) -> A` which provides the optimal action to take for a given state. At this point I would like to hightlight I am strictly talking about a determnisitc domain. The goal state `s8` is not going to change for the agent until it solves the domain. We need this policy to keep providing an action `a` that can maximise our expectation of getting the highest cumulative reward. This is expressed as

<img src="https://latex.codecogs.com/gif.latex?E_{p(s_1:T, a_1:T)}[\sum^T_{t=0} \gamma^t R(s_t, a_t)|\pi] "
style="display: block; 
           margin-left: auto;
           margin-right: auto;"/>

In order to achieve this we use the Bellman equations for **state value function** and **action value function**. To find an optimal policy the following equations need to be solved

<img src="https://latex.codecogs.com/gif.latex?\pi(s) = argmax \{R(s,a) + \gamma \sum_{s'} P(s' | s, a)V(s') \} "
style="display: block; 
           margin-left: auto;
           margin-right: auto;"/>

<img src="https://latex.codecogs.com/gif.latex?V^*(s) = max \{R(s,a) + \gamma \sum_{s'} P(s' | s, \pi(s))V^*(s') \} "
style="display: block; 
           margin-left: auto;
           margin-right: auto;"/>

here `V*(s)` is the optimal state value function. This can be used to derive the optimal action value function 

<img src="https://latex.codecogs.com/gif.latex?Q^*(s,a) = \gamma \sum_{s'} P(s' | s, a)V^*(s') \} "
style="display: block; 
           margin-left: auto;
           margin-right: auto;"/>

In this blog post we will not go into further details of how these Bellman equations were derived nor how they can be solved. I will briefly introduce **Value iteration** and **Q-learning** algorithm.

Insert algorithm for Value iteration and Qlearning. 

