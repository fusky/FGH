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
        
        command = f"{self.joern_path} --script export_graphs.sc --params input={source_code_path}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        graphs = self._parse_joern_output(result.stdout)
        return graphs 

    def build_fusion_graph(self, ast_graph, cfg_graph, ddg_graph) -> nx.Graph:
        
        FG = nx.DiGraph()

        for G in (ast_graph, cfg_graph, ddg_graph):
            for node, attributes in G.nodes(data=True):
                if node not in FG:
                    FG.add_node(node, **attributes)
                else:
                    FG.nodes[node].update(attributes)

        for G in (ast_graph, cfg_graph, ddg_graph):
            for u, v, attributes in G.edges(data=True):
                if FG.has_edge(u, v):
                    FG.edges[u, v].update(attributes)
                else:
                    FG.add_edge(u, v, **attributes)

        for node in FG.nodes():
            ast_feat = ast_graph.nodes[node].get('features', None) if node in ast_graph.nodes else None
            cfg_feat = cfg_graph.nodes[node].get('features', None) if node in cfg_graph.nodes else None
            ddg_feat = ddg_graph.nodes[node].get('features', None) if node in ddg_graph.nodes else None
            
            fused_feature = self._fusion_function_R(ast_feat, cfg_feat, ddg_feat)
            FG.nodes[node]['x'] = fused_feature

        return FG

    def _fusion_function_R(self, *features):
        
        valid_features = [f for f in features if f is not None]
        if not valid_features:
            return None
        return np.mean(valid_features, axis=0)

    def _parse_joern_output(self, stdout: str) -> Dict:
        try:
            data = json.loads(stdout)
        except Exception:
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