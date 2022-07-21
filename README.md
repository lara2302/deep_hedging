# Deep Hedging - A Novel Approach to Hedging Jump Risk
## Repository for the WU Master Thesis by Lara Hofmann

The structure of the repository is as follows:

* **colab** contains the execution scripts for several numerical experiments created with `Google Colaboratory` and includes some of the resulting plots.
* **deep_hedging** contains the functions that define the deep hedging model as well compute the likelihood ratio weights for the application of importance sampling.
* **instruments** contains functions that price different options in different models or define the payoff function:
  * **EuropeanCall** = European standard call in the Black-Scholes model,
  * **MertonJumpCall** = European standard call in the Merton jump model (infinite sum closed-form solution),
  * **MertonJumpPut** = European standard put in the Merton jump model (infinite sum closed-form solution),
  * **payoff_barrier** =  terminal payoff of an up-and-out barrier option (either call or put).
* **loss_metrics** contains different loss metrics that can be used as the loss function of the deep neural network: 
  * **entropy** = entropy (exponential) risk measure,
  * **expectedshortfall** = expected shortfall of the terminal hedging error (only penalizes negative but not positive deviations from zero),
  * **variance_optimal** = variance optimal loss (equals the optimality criterion of quadratic hedging),
  * **variance_optimal_with_ES** = method of Lagrange multipliers that minimizes the variance optimal loss with expected shortfall as constraint.
* **stochastic_processes** contains functions that generate sample paths from either the Black-Scholes or Merton jump diffusion model.
* **traditional_hedging** contains functions that solve a PIDE for either standard European or barrier options in the Merton jump model and compute the hedging strategy for a given set of sample paths using delta, delta-gamma and quadratic hedging.
* **utilities** contains a function that splits simulated data into training and testing sample.
<br> <br>


The bases for the implementation of the deep hedging algorithm and the solution of the Merton PIDE via finite differences are:

1) [Deep hedging implementation](https://github.com/YuMan-Tam/deep-hedging) by Yu Man Tam (last update in Jan 2021),

2) [Merton PIDE](https://github.com/cantaro86/Financial-Models-Numerical-Methods/tree/master/functions) by Nicola Cantarutti (last update in Jul 2020).

To be more specific, the following parts were used and changes were made.

* Deep hedging algorithm (changes made in 2022)
  * **colab** used as a basis for experiments but modified and added a lot.
  * **deep_hedging** utilized, extended from one possible hedging instrument to arbitrary many possible hedging instruments and changed transaction costs and importance sampling.
  * *EuropeanCall* in **instruments** used only for primary functionality tests, afterwards not needed since it is based on BS model and Merton model was required.
  * Only used *entropy* in **loss_metrics** which, however, did not work well for model with jumps and as a consequence eventually not used.
  * *BlackScholesProcess* in **stochastic_processes** used only for primary functionality tests, afterwards not needed since it is based on BS model and Merton model was required.
  * *train_test_split* in **utilities** used without changes.
 
 * Merton PIDE (changes made in 2022)
   * *PIDE_price* in **Merton_pricer** utilized and extended to including barrier options. The output is then used for the implementation of delta, delta-gamma and quadratic hedge.
   * *Option_param* in **Parameters** used for **Merton_pricer** (simply passes on information of respective option).
   * *Merton_process* in **Processes** used for **Merton_pricer** (simply passes on information of respective Merton process).
<br> <br>

**References:**

* David Applebaum. *Levy processes and stochastic calculus.* Cambridge university press, 2009.
* Hans Buehler, Lukas Gonon, Josef Teichmann, and Ben Wood. *Deep hedging.* Quantitative Finance, 19(8):1271-1291, 2019.
* Nicola Cantarutti. *Numerical study of the merton pide in option pricing.* Available at SSRN 3579408, 2020.
* Nicola Cantarutti. *Option pricing in exponential Levy models with transaction costs.* PhD thesis, Universidade de Lisboa (Portugal), 2020.
* RamaCont, PeterTankov, and Ekaterina Voltchkova. *Hedging with options in models with jumps.* In Stochastic analysis and applications, pages 197-217. Springer, 2007.
* Laetitia Badouraly Kassim, Jerome Lelong, and Imane Loumrhari. *Importance sampling for jump processes and applications to finance.* arXiv preprint arXiv:1307.2218, 2013.
* Robert C Merton. *Option pricing when underlying stock returns are discontinuous.* Journal of financial economics, 3(1-2):125-144, 1976.
* Peter Tankov. *Financial modelling with jump processes.* Chapman and Hall/CRC, 2003.
