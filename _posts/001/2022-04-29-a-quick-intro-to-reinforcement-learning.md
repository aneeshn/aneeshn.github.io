---
layout: post
title: "A quick intro to Reinforcement Learning"
date: 2022-04-29 15:58:59 -0200
categories:
  - Research
tags:
  - Reinforcement Learning
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

In this post, I am going to briefly introduce Reinforcement Learning (RL). This is the first of a series of posts that I am trying to tie to my research interests. Starting with this post I will briefly introduce the basic concepts of RL. Picture a world where an Agent (a robot or human being) needs to interact with the world to achieve some goal. It takes some action that it thinks will lead to it achieving the goal. When the agent does achieve the goal it gets a reward. Every action either results in a very high reward or low reward. When you take an action you also tend to change the state of the world. We now depict what an agent needs to be aware of to interact with an unknown world and learn to interact intelligently.  

<figure>
<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 70%;"
    src="/assets/agentworld.png" 
    alt="Simple Grid world"/>
<figurecaption style="display: block; 
           margin-left: auto;
           margin-right:auto;
           text-align: center;
           font-size: 14px;">Fig 1: Agent taking an action to learn and behave intelligently</figurecaption>
</figure>

Picture a simple $$3 \times 3$$ grid world and in this grid world an agent can only move in the cardinal directions: $$\uparrow up, \downarrow down, \leftarrow left$$, and $$\rightarrow right$$. There is only one **goal state** the agent needs to reach, by reaching this goal state the agent has achieved its objective. The agent is interacting with the **grid world**, this is also known as the **environment**. This environment has 9 **states**, starting from $$S_0, S_1, ..., S_8$$. When the agent takes an **action**, for example, let's assume the agent takes the action **down** in the state $$S_5$$, this will move the agent to state $$S_8$$. When the agent reaches state $$S_8$$ it will receive some **reward** $$r$$. This behavior of the environment is known as the **model** of the environment. The model defines the reward and state transition probability. In this case, we are aware of this model, but in many realistic scenarios, we may not know how the model performs.

<figure>
<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"
    src="/assets/gridworld.png" 
    alt="Simple Grid world"/>
<figurecaption style="display: block; 
           margin-left: auto;
           margin-right:auto;
           text-align: center;
           font-size: 14px;">Fig 2: Simple Grid World</figurecaption>
</figure>

Having a model allows us to search and plan to solve the problem. One can use classical search and planning algorithms to solve this. I am interested in using RL to solve this domain. RL can deal with complex and unstructured observations, have some robustness to stochastic environments and not rely on hand-crafted models and optimizations.

In RL we can split them into two categories namely:
- ***Model-based RL***: Algorithms that rely on learning a model of the environment or a model directly to learn. 
- ***Model-free RL***: Algorithms that learn to solve without a dependency on the model of the environment.

### Model
A model describes what the environment should be. When an actor takes an action in the environment it will usually result in a state change and an immediate reward. The environment also needs to inform the actor if the goal was achieved or not. In literature, we use a **transition function** $$P$$ and a **reward function** $$R$$. So what does a transition mean? If the agent is in a state $$s$$ and takes an action $$a$$, this will result in a new state $$s'$$ and a reward $$r$$. This could be represented as a tuple: $$t = <s, a, s', r>$$. 

In a deterministic world, this would be enough, but what if we want to consider a stochastic transition in the environment? What if when you are in some state $$s_5$$ and took action $$\downarrow$$ and instead of always resulting in $$s_8$$; with a 10% probability you ended up in state $$s_4$$ and 90% probability you end up in $$s_8$$ ? Let us assume the floor is slippery and the agent wants to go down one cell but slips with a 10% probability to the left.   

You can represent this as:

$$P(s'=s_8|s=s_5, a=\downarrow) = 0.9$$

$$P(s'=s_4|s=s_5, a=\downarrow) = 0.1$$

Now we want to add the reward to the mix of this. What if we got a reward of 10 when we went to state $$s_8$$ and we got a reward of -1 when we went to state $$s_4$$.

$$P(s'=s_8, r=10|s=s_5, a=\downarrow) = 0.9$$

$$P(s'=s_4, r=-1|s=s_5, a=\downarrow) = 0.1$$

We can express this in a general way: $$P(s',r\vert s, a)$$. To Express the reward we can rewrite the equation as:

$$R(s,a) = \sum_{r\in R} r \sum_{s' \in S} P(s'|s, a)$$

For those of you who remember what an [expectation](https://www.probabilitycourse.com/chapter3/3_2_2_expectation.php) is, the reward function is indeed the expectation of the reward. The equation above can also be rewritten as: $$R(s,a) = \Bbb E[ R_{t+1}\vert s_t = s, a_t =a]$$. 

If you are wondering why we are taking the expectation of the reward instead of the immediate reward itself, it is because life is uncertain. So, for each action, we calculate the average of all possible rewards, weighted by the likelihood of achieving them. From the example above, for the state $$s_5$$ taking the action $$\downarrow$$ results in an average reward of 8.9.

### Policy
When the agent takes a series of actions in states, following a sequence of actions in a state is known as a policy. A function that can generate the action to take in a state is known as a policy function.

### Utility Function
The drawback of only having a reward function, it simply captures the immediate short-term consequences of executing a policy. What if we want to learn the consequences of executing a policy over a longer term?

### Markov Decision Process
For this post I am interested in using Reinforcement Learning (RL) to solve this domain. Lets use the Markov Decision Process (MDP) framework to solve the RL problem. An MDP can be written as a tuple:
$$M = <S, A, P(s'| s, a), R(s, a)>$$
where 

- a set of States $$s ∈ S$$
- a set of Actions $$a ∈ A$$
- an action dependent state transition probablity function $$P(s'\vert s, a)$$
- a reward function $$R(s, a) \rightarrow ℝ$$

If the agent naively moves from one state to another, taking random actions in every state it is in, it will eventually land up in the goal state causing for it to end its "turn". When the agent does this it has generated a sequence of states and actions, this known as a **trajectory**. 

What if we would like to know if there are specific trajectories that are more important than others? In order to do this we need a way to "score" these trajectories. If we sum up all the reward we get from each state until the trajectory ends then that should be good enough right? What if this sequence is really long? We should not assign credit from a state far into the sequence to impact the score of the trajectory. We should probably give more weight to the rewards closer to the beginning to the sequence than further back. I am trying not to go into further detail, I encourage the reader to do some research into why this is the case. So, we define a cumulative reward function as 

$$G = \sum^T_{t=0}\gamma^t R(s_t, a_t)$$

In order to solve an MDP we need to find an optimal policy function $$π(s) -> A$$ which provides the optimal action to take for a given state. At this point I would like to hightlight I am strictly talking about a determnisitc domain. The goal state $$s8$$ is not going to change for the agent until it solves the domain. We need this policy to keep providing an action $$a$$ that can maximise our expectation of getting the highest cumulative reward. This is expressed as

$$E_{p(s_1:T, a_1:T)}[\sum^T_{t=0} \gamma^t R(s_t, a_t)|\pi] $$

In order to achieve this we use the Bellman equations for **state value function** and **action value function**. To find an optimal policy the following equations need to be solved

$$\pi(s) = arg\max_{a} \{R(s,a) + \gamma \sum_{s'} P(s' | s, a)V(s') \}$$

$$V^*(s) = \max_{a} \{R(s,a) + \gamma \sum_{s'} P(s' | s, \pi(s))V^*(s') \}$$

here $$V*(s)$$ is the optimal state value function. This can be used to derive the optimal action value function as

$$Q^*(s,a) = \gamma \sum_{s'} P(s' | s, a)V^*(s') \}$$

which can be written out by expanding $$V^*(s) $$ as:

$$Q^*(s,a) = \gamma \sum_{s'} P(s' | s, a)V^*(s') \}$$

In this blog post we will not go into further details of how these Bellman equations were derived nor how they can be solved.
