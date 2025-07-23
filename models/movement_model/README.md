# Movement


Learning from movement is a tricky process. My favorite way to demonstrate this is with a thought experiment: imagine a world where there’s a big Garden of Eden in the center, surrounded by a vast ring of desert. If you drop someone into the desert, they’ll either choose a random direction and start walking, or they’ll just wander aimlessly. That’s because they don’t know where they are, and everything around them is equally bad — there’s no reason to prefer one direction over another.

Now, if you drop someone into the Garden of Eden, what's amusing is that they’ll do the exact same thing — but for the opposite reason. Everything around them is equally good, so again, there’s no reason to prefer one thing over the next. They’ll either pick a direction at random or wander.

This is the first lesson of studying movement: just because a creature doesn’t show a clear preference for where it is or where it’s going doesn’t mean it’s in a place that’s neither good or bad. It might just mean that, among the options available or known to it, none are clearly better than the rest. So, we need to look for something else to understand where things are likely to end up.

To see this, imagine one of our wanderers approaching the edge of the Garden of Eden. Suddenly, their behavior changes. Instead of wandering randomly, they will reliably avoid the desert. That change is telling.

Understanding how animals behave — where they prefer to be and how they get there — is all about looking for places where movement diverges from simple diffusion.

There are two main ways this can happen.

The first and most obvious is directivity. At the edge of Eden, creatures will begin to move purposefully, pointing themselves back toward the center. Directivity is very informative for understanding population behavior.

So how do you find it?

If you have a movement model — a model that gives you the probabilities of moving in different directions — you can take each point in your space, allow it to diffuse, calculate the center of mass of the diffused creature(s), and then create a vector from the original point to that new center.

If the movement is purely diffusive, the center of mass won’t shift. But if it does, that shift gives you a measure of directivity.

To visualize this on a map, pick a direction of interest, take the dot product of that direction with the movement vector at each point, and plot it. High values indicate strong movement in that direction; low or negative values indicate movement in the opposite direction or orthogonal to it.

By doing this, you’d start to see the edges of Eden. As you rotate the direction of interest, you'd see a kind of sweeping arc — like a semicircle — rotating around the Garden.

The second form of divergence is change in diffusion speed — whether creatures speed up or slow down in certain areas. In other words, whether a region is repulsive or sticky.

To detect this, you can again allow diffusion to happen and then see how much stayed near the original point compared to what you'd expect from pure diffusion.

If more remained than expected, the area is sticky — creatures tend to stay. If less remained, the area is repellent — they tend to leave.

This is useful even if there’s no strong directivity. A place where diffusion slows down will tend to collect things — creatures go in, but they don’t leave as quickly. You can map this too.

By combining these two insights — directivity and changes in diffusion speed — you can capture most of what movement divergence looks like and learn a great deal about your creature.



## Thinking about Information

Our movement model manifests as a matrix $M$ where each row represents a destination in our spatial grid and each column an origin. That is each of the entries $m_{ij}$ represents the likelihood of going from cell $j$ to cell $i$. Therefore if $\rho_t$ is a column vector representing the density across our spatial grid, $\rho_{t+1}=M\rho_t$ is the expected density in the following timestep. 

What can $M$ tell us? Well first suppose that each cell $j$ has some set of neighbors $N$ and that the likelihood of moving to one of these neighbors is equal. That is $m_{ij}\equiv |N|^{-1} \space \forall \space i \in N $. Then if we were to drop a creature (whatever is moving) into cell $j$ our knowledge of it would spread evenly across the space. In some sense this is equivalent to saying the creature has absolutely no preferences about the space and will just randomly distribute itself. This can be thought of as a null diffusion hypothesis (NDH). If this is the reality then there is really nothing anyone can actually say about positioning of the species. For any $M$ we can compute an equivalent $N$ that represents the NDH. 

Clearly if $M$ is going to be of any use to us it has to be different than our $N$. Indeed the question of what we can learn from $M$ is really a question of in what ways can $M$ be different than $N$. 

### Divergence

Each column of $M$ represents the expected transmission of "information" about our creature from a specific grid cell to all other cells. That is for a cell $j$, $m_{ij}$ is actually a probability distribution across the rest of the space. $n_{ij}$ is the same distribution but just for our null diffusion hypothesis. 

We can use the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) to actually measure how much more information $m_{ij}$ has as opposed to $n_{ij}$:

$$D_j = \sum_i m_{ij}\ln{(m_{ij}/n_{ij})} $$

This will be a positive quantity that gets bigger cell $j$'s actual diffusion gets more and more different as compared to the NDH's expected diffusion. 

But how do we go about interpreting this? Well image plotting the $D_j$ across our spatial grid. In the places where it is low we are effectively being told that creatures are more or less behaving as if they are exhibiting a random walk - there are no real preferences about the area if that's the case. However in cases where $D_j>>0$ we know this an area where something interesting is going on. At this point we're not sure exactly what, but at least we know it's worth looking. 

### Stickiness

The first, and simplest, reason as to why we might end up with a divergence would be because creatures are more likely to stay put than diffuse at all. In other words $m_{jj} > m_{ij}$. 

This is a pretty useful thing to know about in the case of fisheries because it can make for some interesting dynamic management. For example suppose that you are wanting to avoid catching a specific species but one day you pull up the long lines and the fish you're trying to avoid is all over 'em. Bummer. Thing is if the fish move around enough it may not be worth avoiding this spot in the future. And how quickly the fish move away from an area is key to understanding that timeline. If they move away slowly - $m_{jj}$ is high - then the area should probably be avoided for a while. If it is low then it probably doesn't matter after a couple of time steps. 

Therefore one thing we can measure from $M$ is just its stickness across the spatial grid. 

### Local Sinks

Next we can think about another local property. For each destination cell $i$ the $m_{ij}$ represent the contribution from origin cell $j$. But consider $\sum_j m_{ij}$ In an infinite spatial grid (i.e. has no boundaries) with simple diffusion we wouldn't expect mass to accumulate anywhere. So if the whole grid was seeded with a $1$ per cell we'd expect a $1$ per cell. That is we'd expect $\sum_j n_{ij}\equiv 1$. Therefore if $\sum_j m_{ij}>1$ for some $i$ we're saying that $i$ is acting as a sink - mass tends to accumulate there. 

In general then:

$$S_i = \ln{\left( \frac{\sum_j m_{ij}}{\sum_j n_{ij}} \right)}$$

Gives us a good sense of where the local sinks are. If this quantity is positive, we're getting more accumulation than we'd expect from our null hypothesis and if it's negative, creatures are (locally) avoiding the area. 

Clearly if one of these exists it means that we'd expect even with a random distribution of creatures for an accumulation or avoidance dependending on whether $S_i$ is positive or negative respectively. 

### Directionality

- [ ] What about the average ratio between source and sink (i.e. break up the sum above)
- [ ] I want one that can deal with directionalities  