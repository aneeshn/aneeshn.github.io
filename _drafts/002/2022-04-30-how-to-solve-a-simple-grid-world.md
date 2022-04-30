---
layout: post
title: "How to solve a simple grid world"
date: 2022-04-30 15:58:59 -0200
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
## Things to write about

- Add how to solve the grid world. 
- What does the value function result to and what does the action value function result to. 
- You also need to add the reward matrix and what that looks like. 
- Is there any pattern in the reward matrix? Introduce the model features paper here
- Introduce the columnar grid world
- Show graphs were the policy is to simple go right
- Show what happens in a 5x5 grid world
- Show what happens when the goal columns are changed around
- mention the drawback of that example, it is too simple being a columnar grid world. Ask the question can it work for a single state?
- Show what that looks like for the 9 different states, and what the action value function looks like for each of those states
- Show if there are any clustering found there?
- Once again bring up the point that it is still too simple 
- Introduce barriers, mention why adding barriers makes the domain more complex.
- Show examples of 3 barriers in a 3x3 grid world. 
- Show the groupings for each of the 3 barrier scenario
- Show how the reward matrix doesn't really change.
- Show what the hypothetical reduced state space looks like


<!-- ### Abstract

In this post I am trying to answer the question `can I learn in a compressed state space?`

What if we are able to reduce the state space we need to explore and learn in ? If so can we transfer the skills we have learnt in this small domain and apply it to a more complex domain? Before I explain why this is important, let me illustrate the point that I am trying to make. -->