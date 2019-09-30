# Adversarial Examples

> make a fool out of deep learning models

## Introduction
The classic example of adversarial attack is the following image:

|![](https://i.imgur.com/EUFEkCh.png) |
|:--:| 
| *From Explaining and Harnessing Adversarial Examples by Goodfellow et al.* |

It is argued by Ian Goodfellow that this is caused by the Linearity of Deep Learning models. The small perturbation in every pixel adds up to become large differences in the outcome. 

[Attack methods](basic_attack.md):

#### norm bounded attacks
l0 norm: # of different pixels
l1 norm: Sum (x_i - y_i)
l2 norm: Sqrt(Sum(x_i^2 - y_i^2))

l_infinity norm: max(x_i - y_i)

* [Fast Gradient Sign Method](basic_attack.md/#FGSM)
* [Projected Gradient Descent](basic_attack.md/#PGD)

Certified Defenses:
* [Convex Polytope](convex_polytope.md)

#### todos
>paper review