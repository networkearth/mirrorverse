**Lead:** Pacific salmon (Oncorhynchus spp) fuel food webs that sustain both wildlife and human communities throughout the North Pacific.
**Development:**
- Born in freshwater, these fish amass 99% of their body weight in the marine environment and when they return to spawn and die, that marine bounty becomes fertilizer for their natal watersheds.
- Those that don't return feed killer whales, Steller sea lions, salmon sharks, and other marine predators.
- For human communities, Pacific salmon have provided nourishment and shaped cultures for millennia.
- In Alaska alone, Pacific salmon constitute one third of the 34 million pound annual subsistence harvest and hundreds of millions of pounds in commercial landings.

**Lead:** Unfortunately these populations are now destabilizing across the North Pacific, threatening the ecosystems and economies they sustain.
**Development:**
- Of the 53 populations of Pacific salmon in the continental United States, more than half are listed under the Endangered Species Act, and despite billions of dollars in recovery programs, none have been delisted.
- Across North America, return rates of Chinook salmon have declined threefold since 1978, while Chum salmon returns in British Columbia's Central Coast region fell by 91% between 1960 and 2020.
- The Yukon River watershed saw Chinook salmon harvests drop 70% between 1998-2010 compared to the prior three decades.
- Sockeye salmon in the Fraser River hit such a low in 2009 that the Prime Minister of Canada established a committee to investigate; that return numbered 900,000 fish, yet by 2019 the return fell to 490,000 and by 2020 to just 290,000.

**Lead:** The picture grows more concerning when viewed through the lens of diversity.
**Development:**
- Between 1970 and the 2010s, Pacific salmon harvests actually increased 4.9-fold in Russia and 2.6-fold in the United States, indicating an overall drop in diversity when combined with the aforementioned declines.
- In Puget Sound, Pink and Chum salmon made up approximately half of the stock in the 1970s; between 2005 and 2016 they made up 80%.
- The stabilizing effect of multiple populations - the portfolio effect - has weakened dramatically in the Skeena River, with returns that were twice as stable as a single population between 1913-1923 dropping to only 1.1 times as stable by 2010-2017.
- Hatcheries, an early response to these declines, have not reversed the trend; of 102 papers reviewed on genetic diversity in populations with hatchery fish, 66 indicated declines.

**Lead:** Pacific salmon are also getting smaller.
**Development:**
- Chinook salmon in Alaska were 8% smaller in samples after 2010 than in samples before 1990, with Coho, Chum, and Sockeye showing reductions of 3.3%, 2.4%, and 2.1% respectively.
- Part of this decline in size reflects a decrease in age at maturity; before 1975, Alaskan Chinook salmon that had spent five years at sea represented 3-5% of returning spawners, but by 2009 that proportion had dropped below 0.5%.
- This shift is especially concerning because larger, older fish are disproportionately important to stock productivity.

**Lead:** Unraveling these population dynamics depends on understanding Pacific salmon's marine ecology as all species of Pacific salmon make extensive use of the marine environment to power their growth.
**Development:**
- Chum salmon migrate to coastal waters immediately after emerging from their gravel beds, rear as juveniles, then range through the open ocean for up to six years before returning to spawn.
- Sockeye salmon spend more time in freshwater (1-3 years) but then follow a similar trajectory, spending up to three years at sea.
- Chinook salmon from rivers in Oregon regularly journey to the Gulf of Alaska during their up to four years in the ocean.
- Pink salmon are perhaps the most impressive in their movements, migrating directly to the marine environment upon hatching and then traveling thousands of kilometers in their 18 months at sea.
- While Coho salmon spend less time in the ocean than their relatives, they still manage to amass 99% of their weight in marine waters.

**Lead:** However, our current understanding of their marine distribution has largely depended on sampling through the capture of fish at sea - leaving us with a partial picture.
**Development:**
- Studies of distribution in the North Pacific have come through high seas tagging studies conducted first by the International North Pacific Fisheries Commission and continued by the North Pacific Anadromous Fish Commission.
- While these studies have been invaluable in prying open the "black box" of Pacific salmon distribution, the spatiotemporal patchiness of sampling leaves the picture incomplete; since the 1990s, there have been fewer and fewer samples in the eastern North Pacific.
- Closer to the North American coast, Weitkamp et al. used recaptures of tagged fish to map the overall distribution of Coho and Chinook salmon by source region and age class, but because sampling relied on a patchwork of different fisheries, the analysis could only operate at coarse geospatial scales.
- Studies attempting finer resolution have been limited to modeling where fisheries and salmon intersect - a particular problem in areas where Pacific salmon is bycatch and fishers are therefore incentivized to avoid it.

**Lead:** Pop-up satellite archival tags (PSATs) provide one means to remove this dependency on fisheries data.
**Development:**
- Once attached to a fish, PSATs record the data required for environmental geolocation for a set period before releasing from the fish, surfacing, and transmitting their data through satellite communications.
- Because data collection does not require fish recovery, PSAT data is fisheries independent.
- It is also a tool that has already been successfully applied to Pacific salmon.

**Lead:** The data from PSATs, however, have two properties that make them unsuitable for traditional species distribution modeling techniques.
**Development:**
- First, PSATs provide positive-only samples; a lack of records in a specific time and place does not indicate an absence of fish.
- Second, because the data captures individual fish movements, it is highly autocorrelated.
- Methods such as MaxEnt and EcoCast exist for working with positive-only samples, but both approaches implicitly require that positive samples are representative of the underlying distribution - an assumption likely contradicted by both the autocorrelated nature of the data and the extraordinary range of Pacific salmon.
- EcoCast also requires artificial generation of negative samples, introducing data not derived from measurement.

**Lead:** An alternative that has not yet been explored is using the movement data from PSATs to adapt particle tracking techniques.
**Development:**
- Lagrangian particle tracking has already been used to understand distributions of larval dispersal, turtle hatchling distribution, and the movement of plastic debris.
- The approach seeds simulations with particles, projects their movements forward, and identifies areas where particles accumulate - also known as basins of attraction.
- For larvae and plastic, movements are guided passively by oceanic currents, winds, and other abiotic phenomena; Pacific salmon, by contrast, present a unique challenge in that they are active swimmers.
- PSAT data is ideal for this because it captures both positive examples (observed movements) and negative examples (possible but unobserved movements) explicitly, enabling the use of traditional modeling techniques.
- PSAT data can be used to build movement models that can then be used with particle tracking techniques to identify basins of attraction; we will refer to this methodology as species redistribution modeling, or SRMs.

**Lead:** While these basins indicate predicted areas of accumulation, there are other factors affecting fish distribution.
**Development:**
- Migrations create immigration and emigration events that may not deposit fish precisely where they may later preferentially accumulate.
- Preferred areas may also be associated with higher mortality, suppressing expected density levels.
- Therefore, to validate whether the basins identified by SRMs are useful indicators of relatively higher concentration, an independent data source is required.

**Lead:** Here is where fisheries-dependent data can come into play.
**Development:**
- If an SRM's predictions are meaningful, we would expect that areas where fish are predicted to accumulate would generally experience higher catch levels than areas fish are predicted to move away from.
- Passing this test would provide evidence that an SRM has not only identified attractors in fish movement, but hotspots in distribution as well.
- Therefore, running this test requires a case study with both movement data from which to build the SRM and catch data to validate it.

**Lead:** Chinook salmon in the Gulf of Alaska provides such a case study.
**Development:**
- A substantial body of PSAT movement data already exists for Chinook salmon in this region.
- An active groundfish fishery generates Chinook salmon bycatch data suitable for validation.

**Lead:** Therefore our goal is to use this case study to understand whether SRMs can serve as a fisheries-independent means of modeling Pacific salmon distributions.
**Development:**
- We will first develop a movement model using Chinook salmon PSAT data in the Gulf of Alaska.
- We will then use this model to perform species redistribution modeling.
- Finally, we will assess the ability of the resulting basins to differentiate Chinook salmon bycatch risk.
