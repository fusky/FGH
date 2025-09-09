import subprocess
import os
import json
import numpy as np
import networkx as nx
from typing import Dict, List

class FusionGraphGenerator:
    def __init__(self, joern_path: str):
        self.joern_path = joern_path

    def analyze_code(self, source_code_path: str) -> Dict:
        """
        使用Joern分析源代码，返回AST, CFG, DDG的图表示。
        这里是一个伪代码示例，实际中需要调用Joern的脚本或API。
        """
        # 假设有一个Joern的脚本可以输出图的JSON格式
        command = f"{self.joern_path} --script export_graphs.sc --params input={source_code_path}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # 解析Joern的输出（通常是JSON）
        graphs = self._parse_joern_output(result.stdout)
        return graphs # 返回一个包含'AST', 'CFG', 'DDG'键的字典

    def build_fusion_graph(self, ast_graph, cfg_graph, ddg_graph) -> nx.Graph:
        """
        将AST, CFG, DDG融合成一个单一的融合图。
        论文中提到使用一个关系函数 R 来遍历和更新节点。
        """
        FG = nx.DiGraph()

        # 1. 添加所有节点（通过逐图遍历，避免将字典放入set导致不可哈希错误）
        for G in (ast_graph, cfg_graph, ddg_graph):
            for node, attributes in G.nodes(data=True):
                if node not in FG:
                    FG.add_node(node, **attributes)
                else:
                    # 合并已有属性
                    FG.nodes[node].update(attributes)

        # 2. 添加所有边（允许覆盖属性，避免不可哈希集合）
        for G in (ast_graph, cfg_graph, ddg_graph):
            for u, v, attributes in G.edges(data=True):
                if FG.has_edge(u, v):
                    FG.edges[u, v].update(attributes)
                else:
                    FG.add_edge(u, v, **attributes)

        # 3. 应用融合函数 R (论文中的 Algorithm 1, Line 5)
        # 这里R的功能是整合来自不同图的节点信息。
        # 例如，使用GraphCodeBERT初始化节点特征
        for node in FG.nodes():
            # 获取该节点在AST, CFG, DDG中的特征（如果有）
            ast_feat = ast_graph.nodes[node].get('features', None) if node in ast_graph.nodes else None
            cfg_feat = cfg_graph.nodes[node].get('features', None) if node in cfg_graph.nodes else None
            ddg_feat = ddg_graph.nodes[node].get('features', None) if node in ddg_graph.nodes else None
            
            # 使用一个函数（例如MLP或注意力）来融合这些特征
            fused_feature = self._fusion_function_R(ast_feat, cfg_feat, ddg_feat)
            FG.nodes[node]['x'] = fused_feature # 将融合后的特征赋给FG的节点

        return FG

    def _fusion_function_R(self, *features):
        """一个简单的融合函数示例（例如取平均值或加权和）"""
        # 这里应该是一个更复杂的操作，比如用一个小型神经网络
        valid_features = [f for f in features if f is not None]
        if not valid_features:
            return None
        return np.mean(valid_features, axis=0)

    def _parse_joern_output(self, stdout: str) -> Dict:
        """解析 Joern 输出，返回含 AST/CFG/DDG 的字典。容错处理非 JSON 情况。"""
        try:
            data = json.loads(stdout)
        except Exception:
            # 占位：返回空图，便于流程继续跑通
            empty = nx.DiGraph()
            return {"AST": empty, "CFG": empty, "DDG": empty}
        def to_nx(gdesc):
            G = nx.DiGraph()
            for n in gdesc.get("nodes", []):
                nid = n.get("id")
                attrs = {k: v for k, v in n.items() if k != "id"}
                G.add_node(nid, **attrs)
            for e in gdesc.get("edges", []):
                u = e.get("source")
                v = e.get("target")
                attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
                G.add_edge(u, v, **attrs)
            return G
        return {
            "AST": to_nx(data.get("AST", {})),
            "CFG": to_nx(data.get("CFG", {})),
            "DDG": to_nx(data.get("DDG", {})),
        }