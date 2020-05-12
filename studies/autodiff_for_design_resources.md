# Resources for Learning More about Automatic Differentiation for Design

by Peter Sharpe

## Overview

So there are two kind of paths that you can use that both arrive at the whole automatic-differentiation-for-design-optimization thing: a computer-science-like path and a math-and-physics-like path.

The CS-like path basically is: This is how you would efficiently find the derivative of a series of operations  using the chain rule. There's a way to do this "forward" and "backward" - oh and hey, design optimization is a prime use-case for this backwards way.

The Math/Physics-like path is probably most related to adjoint methods for PDEs, though we only occasionally directly use PDEs in our aircraft design optimization work (usually we have some surrogate model that approximates this for speed reasons). In the world of PDEs, an adjoint method that lets you effectively "differentiate backwards" through a PDE by exploiting some properties of residuals when you differentiate the governing equation. That turns out to be very useful for design optimization! Oh, and how would we actually efficiently implement this in code?

I think the best way to learn this is to bark up both trees and try to have them meet somewhere in the middle.

## Resources for the CS-like Path

Start with this blog post:
https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

Then listen to this lecture from Matthew Johnson @ Google:
http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/

Keep going here:
https://youtu.be/twTIGuVhKbQ (Joris Gillis, developer of CasADi)

## Resources for the Math/Physics Path

Start here with an early Martin's presentation:
https://www.pathlms.com/siam/courses/479/sections/678/thumbnail_video_presentations/5169

Then continue by looking through more of Martin's stuff:
http://mdolab.engin.umich.edu/presentations
Martin's stuff is also great because it touches on related topics beyond just pure automatic differentiation: MDO architectures, NLP optimization algorithms, and aircraft design concepts.

To go deeper into the math, follow the fearless Qiqi Wang:
https://www.youtube.com/watch?v=l20FDuv2gL4&list=PLcqHTXprNMIPwiMg7zXg-QsgrDCikCIMl

All this "adjoint" stuff should hopefully help you wrap your mind around what it means to take a reverse-mode derivative ("backpropagation").
