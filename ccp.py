import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import combinations
from itertools import product 
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
import time

def l1_norm(x):
    """computes the L1 norm of a vector x""" 
    return np.sum(np.abs(x))


def l1_subgradient(x):
    """computes the subgradient of the L1 norm at a vector x"""
    return np.sign(x)

def ccp_basis_actions(n, dim, max_iter=100, tol=1e-4, verbose=True, restart=10):
    
    best_points = None
    best_min_dist = -float('inf')
    
    # random restarts multiple times to get the best result 
    for r in range(restart):
        if verbose and restart > 1:
            print(f"Restart {r+1}/{restart}")
            
        # initialize basis actions (randomly in [0, 1]^dim)
        z_k = np.random.rand(n, dim)
        
        # define set of point pairs p
        P = list(combinations(range(n), 2))
        
        # save the previous iteration's min distance to check for convergence
        prev_min_dist = 0
        no_improvement_count = 0
        
        # iterate to solve
        for k in range(max_iter):
        
            min_dist = float('inf')
            min_pair = None
            
            for i, j in P:
                dist = l1_norm(z_k[i] - z_k[j])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)
            
            if verbose:
                print(f"Iteration {k}, Min L1 Distance: {min_dist}")
            
            # check the min distance for significant improvement
            if abs(min_dist - prev_min_dist) < tol:
                no_improvement_count += 1
                if no_improvement_count >= 10:  # if no significant improvement for 10 iterations, consider converged 
                    if verbose:
                        print(f"Converged after {k+1} iterations (no significant improvement in min distance)!")
                    break
            else:
                no_improvement_count = 0

            
                
            prev_min_dist = min_dist


           
            
            # 2. set up convex optimizatio problem
            z = cp.Variable((n, dim))  # decision variable: coordinates of points
            s = cp.Variable()          
            Y = {}                     
            
            for i, j in P:
                Y[(i, j)] = cp.Variable(dim, nonneg=True)  # Y_ij ≥ 0
            
            objective_term = 0
            for i, j in P:
                #compute the L1 norm value and subgradient
                current_diff = z_k[i] - z_k[j]
                current_norm = l1_norm(current_diff)
                subgrad = l1_subgradient(current_diff)
                
                # when subgradient is zero
                if np.all(np.abs(subgrad) < 1e-6):
                    subgrad = np.ones_like(subgrad) / np.sqrt(dim)
                
                # linearization
                linearized_term = current_norm + cp.sum(cp.multiply(subgrad, (z[i] - z[j]) - current_diff))
                objective_term += linearized_term
            
            objective = cp.Minimize(s - objective_term)
            
            # constraints
            constraints = []
            
            # 1. -Y_ij ≤ z^i - z^j ≤ Y_ij
            for i, j in P:
                constraints.append(z[i] - z[j] <= Y[(i, j)])
                constraints.append(z[i] - z[j] >= -Y[(i, j)])
            
            # 2. ∑_(i',j')≠(i,j) 1^T Y_i'j' ≤ s, ∀(i,j) ∈ P
            for i, j in P:
                constraint_term = 0
                for i_prime, j_prime in P:
                    if (i_prime, j_prime) != (i, j):
                        constraint_term += cp.sum(Y[(i_prime, j_prime)])
                constraints.append(constraint_term <= s)
            
            # 3. z^i ∈ D 
            for i in range(n):
                constraints.append(z[i] >= 0)
                constraints.append(z[i] <= 1)
            
            
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            
            if prob.status != cp.OPTIMAL:
                if verbose:
                    print(f"Problem not optimally solved. Status: {prob.status}")
                break
            
        
            z_next = z.value
            
            
            
            # update the iteration point 
            z_k = z_next
            

        # calculate the min L1 distance between point pairs in the final result
        min_dist = float('inf')
        for i, j in P:
            dist = l1_norm(z_k[i] - z_k[j])
            min_dist = min(min_dist, dist)
        
        if verbose:
            print(f"Final min L1 distance for this run: {min_dist}")
        
        # update the best result
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = z_k.copy()
    
    if verbose and restart > 1:
        print(f"Best overall min L1 distance: {best_min_dist}")
    
    return best_points

def fairness_threshold_objective(regrets, gamma=1000):
    """
    计算公平阈值目标函数值: (1/m) * Σ max{r_i + γ, max_j r_j}
    
    参数:
    regrets: 所有玩家的regret值列表
    gamma: 阈值参数，默认为0.1
    
    返回:
    目标函数值
    """
    max_regret = max(regrets)
    objective_value = np.mean([max(r + gamma, max_regret) for r in regrets])
    return objective_value


def generate_initial_weights(n_weights, n_samples=10):
    """
    生成初始权重向量集合（满足单纯形约束：非负且和为1）
    """
    points = []
    for _ in range(n_samples):
        # 生成非负随机数
        x = torch.rand(n_weights)
        # 归一化使和为1
        x = x / x.sum()
        points.append(x)
    return torch.stack(points)


def compute_all_regrets(basis_actions, weights, m, nl, E, s_list, c, a, b, beta_values=None, beta_factor=2.0):
    """
    计算所有玩家在给定权重下的regret值
    """
    all_regrets = []
    
    for player_idx in range(m):
        # 计算当前混合策略下的期望成本
        expected_cost = 0
        for k in range(len(basis_actions)):
            player_cost = compute_player_cost(basis_actions[k], player_idx, m, nl, a, b)
            expected_cost += weights[k] * player_cost
        
        # 计算最佳响应的成本
        best_response_cost, _, _, _ = compute_best_response_cost(
            basis_actions, weights, player_idx, m, nl, E, s_list, c, a, b, 
            beta_values=beta_values, beta_factor=beta_factor, max_iter=30
        )
        
        # 计算regret
        regret = expected_cost - best_response_cost
        all_regrets.append(regret)
        
    return all_regrets

def bayesian_optimization(basis_actions, m, nl, E, s_list, c, a, b, 
                         gamma=1000, max_iter=50, n_initial=10, 
                         tolerance=1e-4, patience=30, verbose=True,beta_values=None, beta_factor=2.0):
    """
    使用贝叶斯优化寻找最优权重向量
    """
    n_basis = len(basis_actions)  

    dtype = torch.float64
    
    
    if verbose:
        print(f"generate{n_initial}initial weight vectors...")
    train_x = generate_initial_weights(n_basis, n_initial).to(dtype)
    
    
    train_obj = torch.zeros(len(train_x), 1, dtype=dtype)
    all_player_regrets = []
    
    for i in range(len(train_x)):
        weights = train_x[i].numpy() 
        
        if verbose:
            print(f"compute initial weight vector {i+1}/{len(train_x)} objective function value...")
            
        
        regrets = compute_all_regrets(basis_actions, weights, m, nl, E, s_list, c, a, b, beta_values, beta_factor)
        all_player_regrets.append(regrets)
        
        

    max_regrets = [max(regs) for regs in all_player_regrets]
    overall_max_regret = max(max_regrets)
    adaptive_gamma = overall_max_regret * 0.1
    gamma = 5000

    train_obj = torch.zeros(len(train_x), 1,dtype=dtype)
    for i in range(len(train_x)):
        obj_value = fairness_threshold_objective(all_player_regrets[i], gamma)
        train_obj[i] = obj_value


    
   
    best_values = []
    best_weights = []
    
    
    bounds = torch.stack([torch.zeros(n_basis), torch.ones(n_basis)])
    
    
    no_improvement_count = 0  
    
    for i in range(max_iter):
        start_time = time.time()
        
        
        model = SingleTaskGP(
            train_X=train_x, 
            train_Y=train_obj, 
            outcome_transform=Standardize(m=train_obj.shape[1]),
            input_transform=Normalize(d=train_x.shape[1])
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        
        best_value = train_obj.min()
        EI = ExpectedImprovement(model=model, best_f=best_value, maximize=False)
        
        
        new_point, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            q=1,
            num_restarts=30,
            raw_samples=150,
            
            equality_constraints=[(
                torch.arange(train_x.shape[1]),
                torch.ones(train_x.shape[1]),
                1.0
            )]
        )
        
        
        new_weights = new_point.reshape(-1).numpy()
        
        
        
        if verbose:
            print(f"\niteration {i+1}:")
            print(f"new weight vector: {new_weights.round(3)}")
        
        
        regrets = compute_all_regrets(basis_actions, new_weights, m, nl, E, s_list, c, a, b)
        all_player_regrets.append(regrets)
        
        
        new_obj = fairness_threshold_objective(regrets, gamma)
        
        
        train_x = torch.cat([train_x, torch.tensor(new_weights, dtype=dtype).reshape(1, -1)])
        train_obj = torch.cat([train_obj, torch.tensor([[new_obj]], dtype=dtype)]) 
        
        
        best_idx = train_obj.argmin()
        best_value = train_obj[best_idx].item()- gamma
        best_weight = train_x[best_idx].numpy()
        
        best_values.append(best_value)
        best_weights.append(best_weight)
        
        if verbose:
            print(f"objective function value: {new_obj:.6f}")
            print(f"current best value: {best_value:.6f}")
            print(f"player's regret: {[f'{r:.6f}' for r in regrets]}")
            print(f"maximum of regret: {max(regrets):.6f} (player {np.argmax(regrets)+1})")
            print(f"iteration runtime: {time.time() - start_time:.2f}seconds")
            print("-" * 50)
        
        
        if i > 0 and abs(best_values[-1] - best_values[-2]) < tolerance:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            
        if no_improvement_count >= patience:
            if verbose:
                print(f" {i+1} iteration no improvement.")
            break
            
    
    plt.figure(figsize=(12, 8))
    
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(best_values) + 1), best_values, marker='o', linestyle='-', color='b')
    plt.xlabel('iteration number')
    plt.ylabel('objective function value')
    plt.title('Objective Function Value vs Iteration Number')
    plt.grid(True)
    
    
    plt.subplot(2, 1, 2)
    plt.bar(range(1, n_basis + 1), best_weights[-1])
    plt.xlabel('basis action')
    plt.ylabel('weight')
    plt.title(f'best weight distribution (objective function: {best_value:.6f})')
    plt.xticks(range(1, n_basis + 1))
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('bo_optimization_results.png')
    if verbose:
        plt.show()
    
    
    print("\noptimization complete!")
    print(f"best weight vector: {best_weights[-1].round(3)}")
    print(f"best objective function value: {best_value:.6f}")
    
    
    final_regrets = compute_all_regrets(basis_actions, best_weights[-1], m, nl, E, s_list, c, a, b)
    print("\neach player's final regret:")
    for i, r in enumerate(final_regrets):
        print(f"player {i+1}: {r:.6f}")
    
    # 7. 比较与均匀权重的结果
    uniform_weights = np.ones(n_basis) / n_basis
    uniform_regrets = compute_all_regrets(basis_actions, uniform_weights, m, nl, E, s_list, c, a, b)
    uniform_obj = fairness_threshold_objective(uniform_regrets, gamma)
    
    print("\n均匀权重 vs 优化权重比较:")
    print(f"均匀权重目标函数值: {uniform_obj:.6f}")
    print(f"优化权重目标函数值: {best_value:.6f}")
    print(f"改进比例: {((uniform_obj - best_value) / uniform_obj * 100):.2f}%")
    
    # 8. 保存最优权重到CSV文件
    np.savetxt('best_weights.csv', best_weights[-1], delimiter=',')
    print("\n最优权重已保存至best_weights.csv")
    
    return best_weights[-1], best_value, final_regrets

def ccp_basis_actions_network(n, m, nl, E, s_list, c, a, max_iter=100, tol=1e-10, initial_points=None, verbose=True, seed=None, restart=1, beta_factor=2.0):
    """
    寻找多样化的网络流量分配基础动作
    
    参数:
    n: 基础动作的数量
    m: 用户数量
    nl: 网络链路数量
    E: 节点-链路关联矩阵
    s_list: 用户需求向量列表 [s1, s2, ..., sm]
    c: 链路容量向量
    max_iter: 最大迭代次数
    tol: 收敛容差
    initial_points: 初始基础动作
    verbose: 是否输出详细信息
    seed: 随机种子(可选)
    restart: 保留参数，主要用于保持接口一致
    
    返回:
    优化后的基础动作
    """
    # 设置随机种子(如果提供)
    if seed is not None:
        np.random.seed(seed)
        
    # 计算每个用户的beta值
    beta_values = compute_beta_values(m, nl, E, s_list, c, a)
    # 每个用户的决策维度
    dim = nl * m  # 总维度 = 链路数 * 用户数
    
    # 初始化点集
    if initial_points is not None:
        z_k = initial_points.copy()
    else:
        # 如果没有提供初始点，则生成随机初始点
        z_k = generate_initial_basis_actions(n, m, nl, E, s_list, c, a, beta_factor)
        
    # 定义点对集合P
    P = list(combinations(range(n), 2))
    
    # 保存前一次迭代的最小距离，用于检查收敛性
    prev_min_dist = -float('inf') 
    no_improvement_count = 0
    
    # 迭代求解
    for k in range(max_iter):
        # 计算当前点集中的最小L1距离
        min_dist = float('inf')
        min_pair = None
        
        for i, j in P:
            dist = l1_norm(z_k[i] - z_k[j])
            if dist < min_dist:
                min_dist = dist
                min_pair = (i, j)
        
        if verbose:
            print(f"Iteration {k}, Min L1 Distance: {min_dist}")
        
        # 检查最小距离是否有显著改善
        if abs(min_dist - prev_min_dist) < tol:
            no_improvement_count += 1
            if no_improvement_count >= 10:  # 如果连续10次迭代无显著改善，认为已收敛
                if verbose:
                    print(f"Converged after {k+1} iterations (no significant improvement in min distance)!")
                break
        else:
            no_improvement_count = 0
            
        prev_min_dist = min_dist
        
        # 设置凸优化问题
        z = cp.Variable((n, dim))  # 决策变量：所有基础动作的坐标
        s = cp.Variable()          # 辅助变量
        Y = {}                     # 辅助变量字典
        
        for i, j in P:
            Y[(i, j)] = cp.Variable(dim, nonneg=True)  # Y_ij ≥ 0
        
        # 构建目标函数
        objective_term = 0
        for i, j in P:
            # 计算L1范数值和次梯度
            current_diff = z_k[i] - z_k[j]
            current_norm = l1_norm(current_diff)
            subgrad = l1_subgradient(current_diff)
            
            # 当次梯度为零时
            if np.all(np.abs(subgrad) < 1e-6):
                subgrad = np.ones_like(subgrad) / np.sqrt(dim)
            
            # 线性化
            linearized_term = current_norm + cp.sum(cp.multiply(subgrad, (z[i] - z[j]) - current_diff))
            objective_term += linearized_term
        
        objective = cp.Minimize(s - objective_term)
        
        # 约束条件
        constraints = []
        
        # 1. -Y_ij ≤ z^i - z^j ≤ Y_ij
        for i, j in P:
            constraints.append(z[i] - z[j] <= Y[(i, j)])
            constraints.append(z[i] - z[j] >= -Y[(i, j)])
        
        # 2. ∑_(i',j')≠(i,j) 1^T Y_i'j' ≤ s, ∀(i,j) ∈ P
        for i, j in P:
            constraint_term = 0
            for i_prime, j_prime in P:
                if (i_prime, j_prime) != (i, j):
                    constraint_term += cp.sum(Y[(i_prime, j_prime)])
            constraints.append(constraint_term <= s)
        
        # 3. 网络流约束：每个基础动作中的每个用户流量必须满足流量守恒
        for i in range(n):  # 对每个基础动作
            for u in range(m):  # 对每个用户
                # 提取该用户的流量向量
                x_u = z[i][u*nl:(u+1)*nl]
                
                # 流量守恒约束：Ex_u = s_u
                constraints.append(E @ x_u == s_list[u])
                
                # 非负流量约束
                constraints.append(x_u >= 0)

                if len(a.shape) > 1:
                    cost_coef = a[u]
                else:
                    cost_coef = a
                    
                # 添加基于beta的用户容量约束
                constraints.append(cost_coef @ x_u <= beta_values[u] * beta_factor)
            
                constraints.append(x_u <= c) 
        
        # 求解问题
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            
            if prob.status != cp.OPTIMAL:
                if verbose:
                    print(f"Problem not optimally solved. Status: {prob.status}")
                break
            
            z_next = z.value
        except Exception as e:
            if verbose:
                print(f"Optimization error: {e}")
            break
        
        # 更新迭代点
        z_k = z_next
    
    # 计算最终结果中点对之间的最小L1距离
    min_dist = float('inf')
    for i, j in P:
        dist = l1_norm(z_k[i] - z_k[j])
        min_dist = min(min_dist, dist)
    
    if verbose:
        print(f"Final min L1 distance for this run: {min_dist}")
    
    # 直接返回优化后的点，不再考虑多次restart的情况
    return z_k




def plot_points_3d(points, title="Points in 3D"):
    
    points = np.array(points)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    
    r = [0, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == 1:  
            ax.plot3D(*zip(s, e), color="gray", alpha=0.3)
    
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=100, depthshade=True)
    
    
    for i, point in enumerate(points):
        ax.text(point[0], point[1], point[2], f"{i}", fontsize=12)
    
    
    min_dist = float('inf')
    min_pair = None
    
    for i, j in combinations(range(len(points)), 2):
        dist = l1_norm(points[i] - points[j])
        if dist < min_dist:
            min_dist = dist
            min_pair = (i, j)
    
    
    if min_pair:
        i, j = min_pair
        ax.plot3D([points[i, 0], points[j, 0]], 
                 [points[i, 1], points[j, 1]], 
                 [points[i, 2], points[j, 2]], 'r--', linewidth=2)
        
        
        mid_point = (points[i] + points[j]) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2], 
               f"min dist: {min_dist:.4f}", fontsize=10, color='red')
    
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    ax.set_title(title)
    
    
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

def generate_initial_basis_actions(n, m, nl, E, s_list, c,a, beta_factor=2.0):
    """
    生成n个满足所有约束的多样化初始基础动作
    
    参数:
    n: 基础动作的数量
    m: 用户数量
    nl: 网络链路数量
    E: 节点-链路关联矩阵
    s_list: 用户需求向量列表 [s1, s2, ..., sm]
    c: 链路容量向量
    
    返回:
    n个满足约束的初始基础动作
    """
    beta_values = compute_beta_values(m, nl, E, s_list, c, a)
    initial_points = []
    dim = nl * m
    
    for k in range(n):
        
        random_obj = np.random.rand(dim)
        
    
        x = cp.Variable(dim)
        objective = cp.Minimize(random_obj @ x)
        
    
        constraints = []

        # 1. 每个用户的流量守恒约束
        for i in range(m):
            # 提取当前用户的流量变量
            x_i = x[i*nl:(i+1)*nl]
            # 添加流量守恒约束
            constraints.append(E @ x_i == s_list[i])
        
        # 2. 所有流量非负约束
        constraints.append(x >= 0)
        
        # 3. 基于beta的用户容量约束
        for i in range(m):
            # 提取当前用户的流量变量
            x_i = x[i*nl:(i+1)*nl]
            
            # 使用该用户的a值计算约束
            if len(a.shape) > 1:
                cost_coef = a[i]
            else:
                cost_coef = a
                
            # 添加约束: a^T x_i <= beta_i * factor
            constraints.append(cost_coef @ x_i <= beta_values[i] * beta_factor)
            
            # 保留原始链路容量约束
            constraints.append(x_i <= c)
        
        
            
        
        # 求解优化问题
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            
            if prob.status == cp.OPTIMAL:
                # 获取最优解
                initial_points.append(x.value)
            else:
                print(f"警告: 初始点 {k+1} 求解状态为 {prob.status}，尝试替代方法")
                # 尝试不同的目标函数
                alt_prob = cp.Problem(
                    cp.Minimize(cp.sum(x)),
                    constraints
                )
                alt_prob.solve()
                
                if alt_prob.status == cp.OPTIMAL:
                    initial_points.append(x.value)
                else:
                    # 如果仍然失败，生成一个简单的随机点
                    # (在实际应用中，这种情况应该很少发生，
                    # 因为问题已经设置为可行)
                    print(f"警告: 无法找到初始点 {k+1} 的可行解")
                    random_point = np.random.rand(dim)
                    initial_points.append(random_point)  # 注意：这可能不满足约束
        
        except Exception as e:
            print(f"初始点生成错误: {e}")
            # 如果出现异常，尝试生成一个简单的随机点
            random_point = np.random.rand(dim)
            initial_points.append(random_point)  # 注意：这可能不满足约束
    
    return np.array(initial_points)
def analyze_basis_actions_by_user(basis_actions, m, nl):
    """分析每个用户在不同基础动作间的流量分配差异"""
    n = len(basis_actions)  # 基础动作数量
    
    # 为每个用户创建L1范数距离矩阵
    user_distance_matrices = []
    
    for u in range(m):
        # 创建当前用户的距离矩阵
        dist_matrix = np.zeros((n, n))
        
        # 计算每对基础动作之间当前用户的L1范数距离
        for i, j in product(range(n), range(n)):
            if i != j:  # 不计算自身与自身的距离
                # 提取用户u在基础动作i和j中的流量向量
                flow_i = basis_actions[i][u*nl:(u+1)*nl]
                flow_j = basis_actions[j][u*nl:(u+1)*nl]
                
                # 计算L1范数距离
                dist = np.sum(np.abs(flow_i - flow_j))
                dist_matrix[i, j] = dist
        
        user_distance_matrices.append(dist_matrix)
    
    return user_distance_matrices


def bpr_cost_function(total_flow, a_i, b_i, lambda_param=0.15, kappa=4):
    
    # 确保输入为numpy数组，便于向量化操作
    total_flow = np.array(total_flow)
    a_i = np.array(a_i)
    b_i = np.array(b_i)
    
    # BPR函数积分计算
    base_cost = a_i * total_flow  # 基本成本项
    congestion_cost = a_i * lambda_param * (total_flow**(kappa+1)) / ((kappa+1) * (b_i**kappa))  # 拥堵成本项
    link_costs = base_cost + congestion_cost

    total_cost = np.sum(link_costs)  

    return total_cost

def compute_player_cost(action, player_idx, m, nl, a, b, lambda_param=0.15, kappa=4):
    """
    计算某个基础动作下特定玩家的成本
    
    参数:
    action: 基础动作（所有玩家的流量决策）
    player_idx: 玩家索引
    m, nl: 玩家数量和链路数量
    a, b: BPR参数
    
    返回:
    玩家的成本
    """
    # 计算每条链路上的总流量
    total_flow = np.zeros(nl)
    for u in range(m):
        user_flow = action[u*nl:(u+1)*nl]
        total_flow += user_flow
    
    # 计算玩家的成本 - 移除多余的求和
    cost = bpr_cost_function(total_flow, a[player_idx], b[player_idx], lambda_param, kappa)
    return cost

def compute_correlated_regret(basis_actions, weights, player_idx, m, nl, E, s_list, c, a, b, beta_values=None, beta_factor=2.0):
    """
    计算特定玩家的相关悔值
    
    参数:
    basis_actions: 基础动作集合
    weights: 混合策略权重向量w
    player_idx: 玩家索引
    m, nl, E, s_list, c: 网络参数
    a, b: BPR成本函数参数
    
    返回:
    玩家的相关悔值
    """
    n_basis = len(basis_actions)
    
    
    expected_cost = 0
    for k in range(n_basis):
        player_cost = compute_player_cost(basis_actions[k], player_idx, m, nl, a, b)
        expected_cost += weights[k] * player_cost
    
    
    best_response_cost, _, _, _ = compute_best_response_cost(
        basis_actions, weights, player_idx, m, nl, E, s_list, c, a, b,
        beta_values=beta_values, beta_factor=beta_factor
    )
    
    
    regret = expected_cost - best_response_cost
    
    return regret

def compute_best_response_cost(basis_actions, weights, player_idx, m, nl, E, s_list, c, a, b, beta_values=None, beta_factor=2.0, lambda_param=0.15, kappa=4, max_iter=100, tol=1e-6):
    """
    使用Frank-Wolfe算法计算给定混合策略下特定玩家的最佳响应成本
    """
    # 如果未提供beta值，则计算
    if beta_values is None:
        beta_values = compute_beta_values(m, nl, E, s_list, c, a)
    
    y_k = basis_actions[0][player_idx*nl:(player_idx+1)*nl].copy()
    iterations = []
    costs = []
    grad_norms = []

    initial_cost = compute_mixed_cost_value(y_k, basis_actions, weights, player_idx, m, nl, a, b, lambda_param, kappa)
    iterations.append(0)
    costs.append(initial_cost)
    
    for k in range(max_iter):
        grad = compute_mixed_cost_gradient(y_k, basis_actions, weights, player_idx, m, nl, a, b, lambda_param, kappa)
        grad_norms.append(np.linalg.norm(grad))
        
        # 获取当前玩家的a值
        if len(a.shape) > 1:
            a_i = a[player_idx]
        else:
            a_i = a
        
        # 使用修改后的线性子问题求解函数
        s_k = frank_wolfe_linear_subproblem(
            grad, E, s_list[player_idx], nl, c, 
            a_i, beta_values[player_idx], beta_factor
        )
        
        gamma = 2.0 / (k + 2)  
        y_next = (1 - gamma) * y_k + gamma * s_k

        cost = compute_mixed_cost_value(y_next, basis_actions, weights, player_idx, m, nl, a, b, lambda_param, kappa)
        iterations.append(k+1)
        costs.append(cost)
        
        if np.linalg.norm(y_next - y_k) < tol:
            break
            
        y_k = y_next
    
    best_response_cost = costs[-1]
    return best_response_cost, iterations, costs, grad_norms

def compute_mixed_cost_gradient(y_i, basis_actions, weights, player_idx, m, nl, a, b, lambda_param=0.15, kappa=4):
    """
    计算混合策略下成本函数对玩家i的流量决策y_i的梯度
    
    参数:
    y_i: 玩家i的当前流量决策 (长度为nl的向量)
    basis_actions: 基础动作集合
    weights: 混合策略权重向量
    player_idx: 玩家索引
    其他参数: 网络和成本参数
    
    返回:
    梯度向量 (长度为nl)
    """
    n_basis = len(basis_actions)
    gradient = np.zeros(nl)
    
    for k in range(n_basis):
        # 提取基础动作k中其他玩家的流量
        other_players_flow = np.zeros(nl)
        for u in range(m):
            if u != player_idx:
                user_flow = basis_actions[k][u*nl:(u+1)*nl]
                other_players_flow += user_flow
        
        # 当玩家i使用流量y_i时的总流量
        total_flow = other_players_flow + y_i
        
        # 修正: BPR成本函数对流量的正确导数
        # 对于链路j: a_i[j] * (1 + λ * (total_flow[j]/b_i[j])^κ)
        link_gradient = a[player_idx] * (1 + lambda_param * (total_flow/b[player_idx])**kappa)
        
        # 加权求和
        gradient += weights[k] * link_gradient
    
    return gradient

def frank_wolfe_linear_subproblem(gradient, E, s_i, nl, c, a_i, beta_i, beta_factor=2.0):
    """
    求解Frank-Wolfe算法的线性子问题，考虑beta约束
    
    参数:
    gradient: 当前梯度向量
    E: 节点-链路关联矩阵
    s_i: 玩家i的需求向量
    nl: 链路数量
    c: 链路容量向量
    a_i: 玩家i的成本向量
    beta_i: 玩家i的beta值
    beta_factor: beta倍数
    
    返回:
    线性子问题的最优解
    """
    # 使用CVXPY求解线性规划问题
    y = cp.Variable(nl)
    
    # 目标: min <gradient, y>
    objective = cp.Minimize(gradient @ y)
    
    # 约束条件
    constraints = [
        # 流量守恒约束
        E @ y == s_i,
        # 非负流量约束
        y >= 0,
        # 基于beta的容量约束
        a_i @ y <= beta_i * beta_factor,
        # 保留原始链路容量约束
        y <= c
    ]
    
    # 求解问题
    prob = cp.Problem(objective, constraints)
    prob.solve()
        
    if prob.status == cp.OPTIMAL:
        return y.value
    else:
        print(f"警告: 线性子问题求解状态为 {prob.status}")
        return generate_feasible_flow(E, s_i, nl, c, a_i, beta_i, beta_factor)

def generate_feasible_flow(E, s_i, nl, c, a_i, beta_i, beta_factor=2.0):
    """
    生成一个满足网络约束的可行流量
    """
    # 使用零目标函数求解可行性问题
    y = cp.Variable(nl)
    objective = cp.Minimize(0)
    
    constraints = [
        E @ y == s_i,
        y >= 0,
        a_i @ y <= beta_i * beta_factor,
        y <= c
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status == cp.OPTIMAL:
        return y.value
    else:
        print(f"警告: 无法找到可行流量! 尝试放宽beta约束")
        # 尝试放宽beta约束
        relaxed_prob = cp.Problem(
            objective,
            [E @ y == s_i, y >= 0, y <= c]  # 移除beta约束
        )
        relaxed_prob.solve()
        
        if relaxed_prob.status == cp.OPTIMAL:
            return y.value
        else:
            print("严重警告: 即使放宽约束也无法找到可行流量!")
            return np.ones(nl) / nl * np.sum(np.abs(s_i)) / 2  
    
def compute_mixed_cost_value(y_i, basis_actions, weights, player_idx, m, nl, a, b, lambda_param=0.15, kappa=4):
    """
    计算给定最佳响应y_i下，玩家i在混合策略中的期望成本
    
    参数:
    y_i: 玩家i的流量决策
    basis_actions: 基础动作集合
    weights: 混合策略权重向量
    player_idx: 玩家索引
    其他参数: 网络和成本参数
    
    返回:
    期望成本值
    """
    n_basis = len(basis_actions)
    total_cost = 0.0
    
    for k in range(n_basis):
        # 创建修改后的基础动作: 将玩家i的流量替换为y_i
        modified_action = basis_actions[k].copy()
        modified_action[player_idx*nl:(player_idx+1)*nl] = y_i
        
        # 计算此修改后基础动作下玩家i的成本
        cost = compute_player_cost(modified_action, player_idx, m, nl, a, b, lambda_param, kappa)
        
        # 加权累加
        total_cost += weights[k] * cost
    
    return total_cost


def main_traffic_network():
    """
    使用CCP方法找到多样化的基础动作
    """
    np.random.seed(42 )  #Set random seed to ensure reproducible results
    
    # 1. 定义网络参数
    nn = 4  # 节点数
    nl = 6  # 链路数
    m = 4   # 用户数
    restart = 10  # 重启次数
    
    # 2. 节点-链路关联矩阵
    E = np.array([
        [1, 1, 0, 0, 0, 0],
        [-1, 0, 1, 0, 1, -1],
        [0, -1, 0, 1, -1, 1],
        [0, 0, -1, -1, 0, 0]
    ])
    
    # 3. 用户需求向量
    s1 = np.array([12, 0, 0, -12])  # 用户1：节点1 → 节点4，需求12
    s2 = np.array([8, 0, -8, 0])    # 用户2：节点1 → 节点3，需求8
    s3 = np.array([0, 10, 0, -10])  # 用户3：节点2 → 节点4，需求10
    s4 = np.array([0, 0, 6, -6])    # 用户4：节点3 → 节点4，需求6
    
    s_list = [s1, s2, s3, s4]
    
    # 4. 链路容量限制
    c = np.array([15, 10, 8, 18, 12, 14])* 3
    
    # 5. 生成初始基础动作
    n_basis = 8  # 基础动作数量
    print("生成初始基础动作...")
    best_points = None
    best_min_dist = -float('inf')
    
    for r in range(restart):
        # 为每次restart设置新种子
        np.random.seed(42 + r * 100)  # 使用足够大的间隔确保随机性
        
        if restart > 1:
            print(f"\nRestart {r+1}/{restart}")
            
        # 为每次restart生成新的初始点
        print(f"Generating initial basis actions for Restart {r+1}...")
        initial_points = generate_initial_basis_actions(n_basis, m, nl, E, s_list, c)
        
        # 保存初始点之间的最小L1距离
        P = list(combinations(range(n_basis), 2))
        min_dist_initial = float('inf')
        for i, j in P:
            dist = l1_norm(initial_points[i] - initial_points[j])
            min_dist_initial = min(min_dist_initial, dist)
        print(f"Minimum L1 distance of initial point set: {min_dist_initial}")
        
        # 对当前restart运行CCP算法
        points = ccp_basis_actions_network(
            n=n_basis, 
            m=m, 
            nl=nl, 
            E=E, 
            s_list=s_list, 
            c=c,
            initial_points=initial_points,  # 使用为当前restart生成的初始点
            max_iter=50, 
            verbose=True, 
            seed=42 + r * 100
        )
        
        # 计算优化后的最小L1距离
        min_dist = float('inf')
        for i, j in P:
            dist = l1_norm(points[i] - points[j])
            min_dist = min(min_dist, dist)
            
        # 更新全局最佳结果
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = points.copy()
            
    print(f"\nBest overall min L1 distance: {best_min_dist}")


    

    
    # 7. 打印优化后的基础动作
    print("\n优化后的基础动作:")
    for i, action in enumerate(best_points):
        print(f"\n基础动作 {i+1}:")
        
        # 打印每个用户的流量分配
        for u in range(m):
            user_flow = action[u*nl:(u+1)*nl]
            print(f"用户 {u+1} 的流量分配: {user_flow.round(2)}")
        
        # 计算并打印总流量
        total_flow = np.zeros(nl)
        for u in range(m):
            user_flow = action[u*nl:(u+1)*nl]
            total_flow += user_flow
        
        print(f"链路总流量: {total_flow.round(2)}")
        print(f"容量使用率: {(total_flow/c*100).round(1)}%")


    print("\nTesting Frank-Wolfe algorithm for best response computation...")
    
    # 设置BPR成本函数参数
    a = np.array([
        [1.0, 2.0, 1.5, 1.0, 2.0, 3.0],  # 用户1
        [1.2, 1.8, 1.4, 1.1, 2.2, 2.8],  # 用户2
        [0.9, 2.1, 1.6, 0.9, 1.9, 3.2],  # 用户3
        [1.1, 1.9, 1.5, 1.0, 2.1, 2.9]   # 用户4
    ])
    
    b = np.array([
        [12.0, 10.0, 8.0, 15.0, 10.0, 12.0],  # 用户1
        [13.0, 9.0, 7.5, 16.0, 11.0, 11.0],   # 用户2
        [11.5, 10.5, 8.5, 14.0, 9.5, 12.5],   # 用户3
        [12.5, 9.5, 7.8, 15.5, 10.2, 11.5]    # 用户4
    ])
    
    
    test_player_idx = 1  
    
    uniform_weights = np.ones(n_basis) / n_basis
    
    
    expected_cost = 0
    for k in range(n_basis):
        player_cost = compute_player_cost(best_points[k], test_player_idx, m, nl, a, b)
        expected_cost += uniform_weights[k] * player_cost
    
    print(f"User {test_player_idx+1} expected cost under uniform mixed strategy: {expected_cost:.4f}")
    
    
    
    best_response_cost, iterations, costs, grad_norms = compute_best_response_cost(
        best_points, uniform_weights, test_player_idx, m, nl, E, s_list, c, a, b, max_iter=50
    )
    
    print(f"User {test_player_idx+1} best response cost: {best_response_cost:.4f}")
    
    
    regrets = [expected_cost - cost for cost in costs]
    
    print("\nRegret values during Frank-Wolfe iterations:")
    for i, regret in enumerate(regrets):
        # Fixed code - calculate conditional expression first, then format
        grad_norm_value = grad_norms[i-1] if i > 0 else 0
        print(f"Iteration {iterations[i]}: Regret = {regret:.6f}, Cost = {costs[i]:.4f}, Gradient Norm = {grad_norm_value:.4f}")
    
    # Plot regret changes over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, regrets, 'bo-')
    plt.xlabel('Iterations')
    plt.ylabel('Regret Value')
    plt.title(f'User {test_player_idx+1} Regret Value Changes during Frank-Wolfe Iterations')
    plt.grid(True)
    plt.savefig('regret_convergence.png')
    plt.show()
    
    return best_points

def compute_beta_values(m, nl, E, s_list, c, a):
    """
    为每个用户计算beta值，即最短路径成本
    
    参数:
    m: 用户数量
    nl: 链路数量
    E: 节点-链路关联矩阵
    s_list: 用户需求向量列表
    c: 链路容量向量
    a: 成本参数
    
    返回:
    每个用户的beta值列表
    """
    beta_values = []
    
    for i in range(m):
        # 为用户i求解最短路径问题
        x = cp.Variable(nl)
        
        # 使用a[i]作为成本系数 (如果a是每个用户的向量列表)
        if len(a.shape) > 1:
            cost_coef = a[i]
        else:
            cost_coef = a
            
        objective = cp.Minimize(cost_coef @ x)
        
        constraints = [
            E @ x == s_list[i],  # 流量守恒
            x >= 0,             # 非负流量
            x <= c              # 容量约束
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        if prob.status == cp.OPTIMAL:
            beta_values.append(prob.value)
            print(f"用户 {i+1} 的最短路径成本 (beta): {prob.value:.4f}")
        else:
            # 如果无法优化，使用一个合理的默认值
            print(f"警告: 用户 {i+1} 的最短路径问题未能求解，使用默认beta值")
            # 使用链路容量的一个分数作为默认值
            default_beta = np.sum(c) * 0.1  
            beta_values.append(default_beta)
    
    return np.array(beta_values)

def sioux_falls_network_test():
    """
    Using Sioux Falls network data to test CCP method and calculate user regrets
    """
    print("Loading Sioux Falls network data...")
    
    try:
        # 1. load incidence matrix E
        E = np.loadtxt('Sioux Falls Network Data - Incidence Matrix E.csv', delimiter=',')
        
        # 2. load nominal cost as BPR parameter a
        a_base = np.loadtxt('Sioux Falls Network Data - Nominal Cost.csv')
        
        # 3. load nominal flow as BPR parameter b
        b_base = np.loadtxt('Sioux Falls Network Data - Nominal Flow.csv')/1000.0
        
        # 4. load demand matrix
        demand_matrix = np.loadtxt('Sioux Falls Network Data - Demand Matrix.csv', delimiter=',')/1000.0
        
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None
    
    
    nn = E.shape[0]  
    nl = E.shape[1]  
    
    print(f"Network size: {nn} nodes, {nl} links")
    
    
    m = 2  # Number of users to select
    
    
    flat_demands = demand_matrix.flatten()
    top_indices = np.argsort(flat_demands)[-m:][::-1]
    
    
    od_pairs = [(0, 15), (11, 1)]
    #for idx in top_indices:
    #    origin = idx // nn
    #    destination = idx % nn
    #    if origin != destination and demand_matrix[origin, destination] > 0:
    #        od_pairs.append((origin, destination))
    #        print(f"User {len(od_pairs)}: Origin {origin+1} to Destination {destination+1}, Demand: {demand_matrix[origin, destination]}")
    #        if len(od_pairs) == m:
    #            break
    
    
    #if len(od_pairs) < m:
    #    print(f"Warning: Only found {len(od_pairs)} valid OD pairs with positive demand")
    #    m = len(od_pairs)
    
    
    s_list = []
    for origin, destination in od_pairs:
        s = np.zeros(nn)
        s[origin] = demand_matrix[origin, destination]  
        s[destination] = -demand_matrix[origin, destination] 
        s_list.append(s)
    
    
    c = b_base * 1.5
    
    
    a = np.array([a_base for _ in range(m)])
    b = np.array([b_base for _ in range(m)])

    beta_values = compute_beta_values(m, nl, E, s_list, c, a)
    beta_factor = 2000.0  
    
    
    print("\nGenerating basis actions using CCP method...")
    n_basis = 10
    restart = 10  
    
    
    best_points = None
    best_min_dist = -float('inf')
    
    for r in range(restart):
        
        np.random.seed(42 + r * 100)  
        
        print(f"\nRestart {r+1}/{restart}")
            
        
        print(f"Generating initial basis actions for Restart {r+1}...")
        initial_points = generate_initial_basis_actions(n_basis, m, nl, E, s_list, c, a, beta_factor)
        
        
        P = list(combinations(range(n_basis), 2))
        min_dist_initial = float('inf')
        for i, j in P:
            dist = l1_norm(initial_points[i] - initial_points[j])
            min_dist_initial = min(min_dist_initial, dist)
        print(f"Minimum L1 distance of initial point set: {min_dist_initial}")
        
        
        points = ccp_basis_actions_network(
            n=n_basis, 
            m=m, 
            nl=nl, 
            E=E, 
            s_list=s_list, 
            c=c,
            a=a,
            initial_points=initial_points,  
            max_iter=30, 
            verbose=True, 
            seed=42 + r * 100,
            beta_factor=beta_factor
        )
        
        
        min_dist = float('inf')
        for i, j in P:
            dist = l1_norm(points[i] - points[j])
            min_dist = min(min_dist, dist)
            
      
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = points.copy()
            
    print(f"\nBest overall min L1 distance: {best_min_dist}")
    
    
    print("\nVerifying diversity of best basis actions:")
    for i in range(n_basis):
        for j in range(i+1, n_basis):
            dist = l1_norm(best_points[i] - best_points[j])
            print(f"L1 distance between basis actions {i+1} and {j+1}: {dist}")
    
    print("\nTesting regret values for different weight distributions...")
    
    
   
    
    
    
    return best_points, s_list, E, c, a, b,beta_values, beta_factor

def test_regrets(basis_actions, weights, m, nl, E, s_list, c, a, b, beta_values=None, beta_factor=2.0):
    """
    Test regret values for all users under given weight distribution
    """
    print(f"Weight distribution: {weights.round(3)}")
    
    all_regrets = []
    
    for player_idx in range(m):
        
        expected_cost = 0
        for k in range(len(basis_actions)):
            player_cost = compute_player_cost(basis_actions[k], player_idx, m, nl, a, b)
            expected_cost += weights[k] * player_cost
        
        
        best_response_cost, iterations, costs, grad_norms = compute_best_response_cost(
            basis_actions, weights, player_idx, m, nl, E, s_list, c, a, b, 
            beta_values=beta_values, beta_factor=beta_factor, max_iter=30
        )
        
        
        regret = expected_cost - best_response_cost
        all_regrets.append(regret)
        
        
        print(f"User {player_idx+1}:")
        print(f"  Expected cost: {expected_cost:.4f}")
        print(f"  Best response cost: {best_response_cost:.4f}")
        print(f"  Regret: {regret:.6f}")
        
        
        if len(iterations) > 2:
            improvement = (costs[0] - costs[-1]) / costs[0] * 100
            print(f"  Cost reduction: {improvement:.2f}% in {len(iterations)-1} iterations")
    
    
    max_regret = max(all_regrets)
    max_regret_player = np.argmax(all_regrets) + 1
    print(f"\nMaximum regret: {max_regret:.6f} (User {max_regret_player})")
    
   
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, m+1), all_regrets)
    plt.xlabel('User')
    plt.ylabel('Regret Value')
    plt.title(f'Regret Values for Different Users (Weight Distribution: {weights[0]:.2f}/{weights[-1]:.2f})')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(1, m+1))
    plt.savefig(f'regrets_w{weights[0]:.2f}.png')
    plt.show()
    
    return all_regrets

def export_user_matrices_to_csv(user_matrices, m, n_basis, filename_prefix="user_distance_matrix"):
    """
    将用户距离矩阵导出为CSV文件，便于上传到Google Sheets
    
    参数:
    user_matrices: analyze_basis_actions_by_user函数返回的用户距离矩阵列表
    m: 用户数量
    n_basis: 基础动作数量
    filename_prefix: 文件名前缀
    """
    import pandas as pd
    import os
    
    # 创建导出目录(如果不存在)
    export_dir = "matrix_export"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # 为每个用户创建并导出一个CSV文件
    for u in range(m):
        # 创建DataFrame，添加行列标签
        df = pd.DataFrame(user_matrices[u])
        
        # 添加列标题(基础动作编号)
        df.columns = [f"Action {i+1}" for i in range(n_basis)]
        
        # 添加行索引(基础动作编号)
        df.index = [f"Action {i+1}" for i in range(n_basis)]
        
        # 保存为CSV
        filename = f"{export_dir}/{filename_prefix}_user{u+1}.csv"
        df.to_csv(filename)
        print(f"用户{u+1}的距离矩阵已保存至: {filename}")
    
    # 创建一个汇总文件，包含所有用户的数据
    all_data = pd.DataFrame()
    
    for u in range(m):
        # 为每个用户创建一个部分，用空行分隔
        if u > 0:
            all_data = pd.concat([all_data, pd.DataFrame([[""] * n_basis])])
        
        # 添加用户标题行
        header_row = pd.DataFrame([[f"User {u+1} Distance Matrix"] + [""] * (n_basis-1)])
        all_data = pd.concat([all_data, header_row])
        
        # 添加用户矩阵数据
        user_df = pd.DataFrame(user_matrices[u])
        user_df.columns = [f"Action {i+1}" for i in range(n_basis)]
        user_df.index = [f"Action {i+1}" for i in range(n_basis)]
        
        all_data = pd.concat([all_data, user_df])
    
    # 保存汇总文件
    summary_filename = f"{export_dir}/all_user_matrices.csv"
    all_data.to_csv(summary_filename)
    print(f"所有用户的距离矩阵已保存至汇总文件: {summary_filename}")

def save_basis_actions(best_points, s_list, E, c, a, b, filename="cached_basis_actions.npz", beta_values=None, beta_factor=2.0):
    """保存基础动作和相关网络数据到文件"""
    print(f"\n保存基础动作和网络数据到 {filename}...")
    
    save_dict = {
        'best_points': best_points,
        's_list': np.array(s_list, dtype=object),
        'E': E,
        'c': c,
        'a': a,
        'b': b
    }
    
    # 如果提供了 beta_values，也保存它们
    if beta_values is not None:
        save_dict['beta_values'] = beta_values
        save_dict['beta_factor'] = beta_factor
    
    np.savez(filename, **save_dict)
    print("保存完成！")

def load_basis_actions(filename="cached_basis_actions.npz"):
    """从文件加载基础动作和相关网络数据"""
    try:
        print(f"\n尝试从 {filename} 加载缓存的基础动作...")
        data = np.load(filename, allow_pickle=True)
        
        best_points = data['best_points']
        s_list = data['s_list']
        E = data['E']
        c = data['c']
        a = data['a']
        b = data['b']
        
        # 尝试加载 beta_values，如果存在的话
        beta_values = data['beta_values'] if 'beta_values' in data else None
        beta_factor = data['beta_factor'] if 'beta_factor' in data else 2.0
        
        print("成功加载缓存的基础动作!")
        print(f"基础动作数量: {len(best_points)}")
        print(f"用户数量: {len(s_list)}")
        print(f"网络规模: {E.shape[0]}节点, {E.shape[1]}链路")
        
        # 返回所有加载的数据，包括 beta_values 和 beta_factor
        return best_points, s_list, E, c, a, b, beta_values, beta_factor
    
    except (FileNotFoundError, IOError):
        print("未找到缓存文件或无法读取，将重新计算基础动作...")
        return None
    

if __name__ == "__main__":
    print("Testing with Sioux Falls network data...")
    
    
    cache_file = "cached_basis_actions.npz"
    loaded_data = load_basis_actions(cache_file)
    
    if loaded_data is not None:
        best_points, s_list, E, c, a, b,beta_values, beta_factor = loaded_data
        beta_values = compute_beta_values(len(s_list), E.shape[1], E, s_list, c, a)
        beta_factor = 2000.0
    else:
        
        best_points, s_list, E, c, a, b,beta_values, beta_factor  = sioux_falls_network_test()
        
        
        if best_points is not None:
            save_basis_actions(best_points, s_list, E, c, a, b, cache_file,beta_values, beta_factor)
    
    if best_points is None:
        print("Error: Failed to load Sioux Falls network data or generate basis actions.")
        exit(1)
    
    m = len(s_list)  
    n_basis = len(best_points)  
    nl = E.shape[1]  
    
    
    
    print("\nstarting Bayesian optimization for weights...")
    best_weights, best_obj, final_regrets = bayesian_optimization(
        basis_actions=best_points,
        m=m, nl=nl, E=E, s_list=s_list, c=c, a=a, b=b,  
        gamma= 1000, 
        max_iter=200,  
        n_initial=8,  
        verbose=True,
        beta_values=beta_values,  
        beta_factor=beta_factor
    )
    
    print("\n use final weights to test regrets...")
    print("\nScenario: Bayesian Optimized Weights")
    test_regrets(best_points, best_weights, m, nl, E, s_list, c, a, b,beta_values, beta_factor)
   
    
    




