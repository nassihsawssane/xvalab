# Applications of XVA &nalysis and numerical Methods

This repository is an academic project developed as part of the course *XVA Analysis and the Embedded Probabilistic, Risk Measure, and Machine Learning Issues* by Stéphane Crépey (M2MO).

It explores selected applications of CVA analysis through their numerical implementation. 
It served as an opportunity to put into practice the methods covered in the course, namely nested Monte Carlo for CVA and neural-network regression for path-wise XVAs and to gain hands-on experience with GPU programming via Numba CUDA. The implementations follow established methods from the literature, and the relevant references are listed below.



## Repository structure

- The two main deliverables are the notebooks `1_nested_cva_estimator.ipynb` and `2_nn_cva_estimator.ipynb` which present respectively the nested Monte Carlo CVA estimator and the neural-network CVA estimator on the IRS and Bermudan swaption books. 
- They both rely on the underlying modules. The product pricers live in `products/` with an `irs/` folder for the closed-form IRS pricer under Vasicek and a `swaption/` folder for the Bermudan swaption pricer via Longstaff-Schwartz each split into a `cpu.py` reference and a `gpu.py` device function. 
- The market simulation and the nested-MC CVA themselves live in `simulation/` again with a CPU reference (`simulation_cpu.py`) and a GPU version (`simulation.py`) together with the timing and confidence-interval helpers in `stats.py`.
- The neural network framework is handled by `cva_nn_estimator.py` which wraps the estimator as the `LearnedCVA` class and by `nn_regressor.py` which provides the generic regressor and training loop with, `utils.py` collecting helpers.
All simulations in the notebooks rely on the GPU versions of these modules. The CPU files serve as references for the consistency checks documented in `gpu_design_notes.ipynb`.
Two additional companion notebooks document specific aspects of the implementation: `gpu_design_notes.ipynb` (GPU design choices and CPU/GPU consistency checks) and `nn_architecture.ipynb` (architecture tuning of the regressor).


## References

<a id="achs"></a>
**[1]** C. Albanese, S. Crépey, R. Hoskinson, B. Saadeddine.
*XVA Analysis From the Balance Sheet.* Quantitative Finance, 21(1), 99–123, 2021.

<a id="acs"></a>
**[2]** L. A. Abbas-Turki, S. Crépey, B. Saadeddine.
*Pathwise CVA Regressions With Oversimulated Defaults.* Mathematical Finance, 2022.

<a id="acd"></a>
**[3]** L. A. Abbas-Turki, S. Crépey, B. Diallo.
*XVA Principles, Nested Monte Carlo Strategies, and GPU Optimizations.* International Journal of Theoretical and Applied Finance, 21(6), 1850030, 2018.

<a id="course"></a>
**[4]** J.F. Chassagneux.  
*Numerical Methods in Financial Engineering.* Course notes, 2026.

<a id="giles"></a>
**[5]** M. Giles.
*Advanced Monte Carlo Methods: American Options.* Lecture notes, Oxford University Mathematical Institute.

<a id="atg"></a>
**[6]** L. A. Abbas-Turki, S. Graillat.
*Resolution of a large number of small random symmetric linear systems in single precision arithmetic on GPUs.* Journal of Supercomputing, 73(4), 1360–1386, 2017.

<a id="ls"></a>
**[7]** F. A. Longstaff, E. S. Schwartz.
*Valuing American Options by Simulation: A Simple Least-Squares Approach.* Anderson Graduate School of Management, Finance, UCLA, 2001.

<a id="aaad"></a>
**[8]** B. Saadeddine. *NeuralXVA: simulation and neural-net learning of path-wise XVAs.* GitHub repository, 2020.

<a id="gpw"></a>
**[9]** M. Germain, H. Pham, X. Warin.
 *Approximation error analysis of some deep backward schemes for nonlinear PDEs.* SIAM Journal on Scientific Computing, 43(5), 2021.

<a id="acss"></a>
**[10]** L. Abbas-Turki, S. Crépey, B. Saadeddine, W. Sabbagh.
 *Pathwise XVAs: The Direct Scheme.* Preprint, 31 October 2022.

<a id="fg"></a>
**[11]** R. Ferguson, A. Green. 
*Deeply Learning Derivatives.* Preprint, version 2.1, 14 October 2018. arXiv:1809.02233.

<a id="markall"></a>
**[12]** G. Markall.
*Numba for CUDA Programmers.* NVIDIA, course materials (5 sessions), 2021.