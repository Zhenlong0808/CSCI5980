# Correlated Equilibrium in Traffic Networks

## Project Overview
This project implements an algorithm for finding diverse basis actions and correlated equilibrium in traffic networks using Convex-Concave Procedure (CCP) and Bayesian optimization methods. The algorithm aims to minimize regret values of users by optimizing a fair threshold objective function, thereby achieving an approximate correlated equilibrium.

---

## Mathematical Background
The project focuses on the following mathematical concepts:

1. **Correlated Equilibrium**: A game-theoretic solution concept that generalizes Nash equilibrium  
2. **Regret Minimization**: Approaching correlated equilibrium by minimizing each user’s regret value  
3. **Network Flow Problems**: Modeling user behavior in traffic networks using flow conservation constraints  
4. **BPR Cost Function**: Function for representing congestion cost in traffic networks  

**Mathematical definition of network flow constraints:**
- User feasible set:  
  \[
  \mathbb{X} \coloneqq \{x \in \mathbb{R}^{n_f} \mid E_ix = f_i, x_i \geq 0\ \forall i, x_i \leq c_i \}
  \]
- Total feasible domain:  
  \[
  \mathbb{D} \coloneqq \prod_{i=1}^m \mathbb{X}_i
  \]

---

## Key Algorithms

1. **CCP (Convex-Concave Procedure)**: Used to generate diverse basis action sets  
2. **Frank-Wolfe Algorithm**: Used to compute users’ best response strategies  
3. **Bayesian Optimization**: Used to find optimal mixed strategy weight allocation  

---

## File Structure

- `ccp.py`: Main code file containing all algorithm implementations
- **Sioux Falls Network Data – `*.csv`**: Sioux Falls traffic network data files
  - `Incidence Matrix E.csv`: Node-link incidence matrix
  - `Nominal Cost.csv`: Initial link costs
  - `Nominal Flow.csv`: Initial link flows
  - `Demand Matrix.csv`: OD demand matrix
- `cached_basis_actions.npz`: Cached basis actions and network parameters
- `best_weights.csv`: Saved optimal mixed strategy weights
- `bo_optimization_results.png`: Visualization of Bayesian optimization results

---

## Main Functions

### 1. Basis Action Generation and Optimization
- `ccp_basis_actions_network()`: Generates diverse basis actions using CCP method
- `generate_initial_basis_actions()`: Generates initial basis actions satisfying network constraints

### 2. User Cost and Regret Calculation
- `compute_player_cost()`: Calculates a user’s cost in a specific basis action  
- `compute_best_response_cost()`: Calculates best response cost using Frank-Wolfe algorithm  
- `compute_correlated_regret()`: Calculates a user’s regret value  

### 3. Bayesian Optimization
- `bayesian_optimization()`: Finds optimal weight vector using Bayesian optimization  
- `fairness_threshold_objective()`: Calculates fair threshold objective function value  

### 4. Testing and Analysis
- `test_regrets()`: Tests regret values for all users under different weight distributions  
- `analyze_basis_actions_by_user()`: Analyzes flow allocation differences between basis actions for each user  

---

## How to Run

### 1. Install Dependencies
```bash
pip install numpy cvxpy matplotlib torch botorch gpytorch pandas
```

### 2. Prepare Data
Ensure Sioux Falls network data files are in the project directory.

### 3. Run Main Program
```bash
python ccp.py
```

---

## Main Execution Flow
- Loads cached basis actions or generates them if not found  
- Generates diverse basis action set using CCP  
- Applies Bayesian optimization to find optimal weight vector  
- Calculates and compares user regret values under different strategies  

---

## Parameter Tuning

- `beta_factor`: Controls strictness of user flow constraints (default: 2000.0)  
- `n_basis`: Number of basis actions (default: 10)  
- `gamma`: Fair threshold parameter for fairness in objective function (default: 1000)  
- `max_iter`: Number of optimization iterations (default: 200)  

---

## Results Interpretation

- **Regret Values**: Lower values indicate strategy is closer to correlated equilibrium  
- **Minimum L1 Distance**: Measures diversity of basis action set, higher is better  
- **Objective Function Value**: Main optimization target; considers maximum regret and fairness  

---

## Research Questions
This project explores:

1. How to effectively generate diverse basis action sets  
2. How to optimize mixed strategy weights to minimize user regret values  
 

---

## Example Results
Running the project will produce:

- Diverse basis action set (flow allocation schemes)  
- Optimal mixed strategy weights  
- Regret values for each user under different strategies  
- Optimization convergence information  
