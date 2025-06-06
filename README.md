# NN2SP

This repository contains Python code developed under the PRIN2020 project:
ULTRAOPTYMAL - Urban Logistics and sustainable TRAnsportation: OPtimization under uncertainTY and MAchine Learning.

Current version of the code addresses two optimization problems:
- Single-Source Capacitated Facility Location Problem (ssCFLP)
- Multi-Path Traveling Salesman Problem (mpTSP)

Neural networks are trained as surrogate models for the recourse problem in two-stage stochastic programming. They learn to predict solution quality across varying scenarios and are subsequently embedded into the original optimization model, thereby reducing the overall model complexity.

## Requirements
Install the following Python packages before running the code:

- pip install gurobipy
- pip install scikit-learn
- pip install gurobi-machinelearning
- pip install pyomo
- pip install mpi-sppy
- pip install tensorflow

Training data can be generated using recourse.py.
The main implementation is in surrogate.py.

## Related Publication
If you find this code helpful in your work, we kindly ask that you cite:
- Chou, X., Messina, E., W. Wallace, S. (2025). Solving Two-Stage Stochastic Programming Problems via Machine Learning. In: Nicosia, G., Ojha, V., Giesselbach, S., Pardalos, M.P., Umeton, R. (eds) Machine Learning, Optimization, and Data Science. LOD ACAIN 2024 2024. Lecture Notes in Computer Science, vol 15508. Springer, Cham. https://doi.org/10.1007/978-3-031-82481-4_1
- Chou, X., Di Marco, L., Messina, E. (2025). Overcoming Computational Challenges in Two-Stage Multi-path Traveling Salesman Problem via Neural Networks. In Proceedings of IES 2025: Innovation & Society: Statistics and Data Science for Evaluation and Quality. Accepted for publication.

