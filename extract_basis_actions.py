import numpy as np
import pandas as pd
import os

def extract_basis_actions_to_csv(npz_file="cached_basis_actions.npz", output_dir="basis_actions_csv"):
    """
    从NPZ文件中提取基础动作向量并保存为CSV格式
    
    参数:
    npz_file: 包含基础动作的NPZ文件路径
    output_dir: 输出CSV文件的目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载NPZ文件
    try:
        print(f"加载文件: {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        best_points = data['best_points']
        
        # 获取基础动作数量和用户数量
        n_basis = len(best_points)
        m = len(data['s_list'])
        nl = data['E'].shape[1]  # 链路数量
        
        print(f"找到 {n_basis} 个基础动作向量")
        print(f"网络信息: {m}个用户, {nl}条链路")
        
        # 创建用户视图文件 - 这是最有用的格式
        for user in range(m):
            user_view_df = pd.DataFrame()
            
            for i, action in enumerate(best_points):
                user_flow = action[user*nl:(user+1)*nl]
                user_view_df[f"Action_{i+1}"] = user_flow
            
            user_view_df.index = [f"Link_{j+1}" for j in range(nl)]
            
            # 只保留非零流量行，简化视图
            nonzero_df = user_view_df.loc[(user_view_df != 0).any(axis=1)]
            
            # 保存用户视图（完整版和非零流量版）
            user_view_file = os.path.join(output_dir, f"user_{user+1}_flow.csv")
            nonzero_view_file = os.path.join(output_dir, f"user_{user+1}_nonzero_flow.csv")
            
            user_view_df.to_csv(user_view_file)
            nonzero_df.to_csv(nonzero_view_file)
            
            print(f"已保存用户{user+1}流量分配: ")
            print(f"  - 完整版: user_{user+1}_flow.csv ({len(user_view_df)}行)")
            print(f"  - 非零流量版: user_{user+1}_nonzero_flow.csv ({len(nonzero_df)}行)")
        
        # 创建原始向量格式（每个基础动作一行）
        raw_vectors_df = pd.DataFrame()
        for i, action in enumerate(best_points):
            raw_vectors_df[f"Basis_Action_{i+1}"] = action
        
        raw_vectors_file = os.path.join(output_dir, "raw_basis_vectors.csv")
        raw_vectors_df.to_csv(raw_vectors_file)
        print(f"已保存原始向量格式: raw_basis_vectors.csv")
        
        # 创建每个链路的总流量分析
        total_flow_df = pd.DataFrame()
        for i, action in enumerate(best_points):
            flow_by_link = np.zeros(nl)
            for user in range(m):
                user_flow = action[user*nl:(user+1)*nl]
                flow_by_link += user_flow
            
            total_flow_df[f"Action_{i+1}"] = flow_by_link
        
        total_flow_df.index = [f"Link_{j+1}" for j in range(nl)]
        total_flow_file = os.path.join(output_dir, "total_flow_by_link.csv")
        total_flow_df.to_csv(total_flow_file)
        print(f"已保存链路总流量分析: total_flow_by_link.csv")
        
        return True
    
    except Exception as e:
        print(f"提取基础动作时出错: {e}")
        return False

# 如果需要直接运行此文件
if __name__ == "__main__":
    extract_basis_actions_to_csv()