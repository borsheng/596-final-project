# src/data_preprocessing.py
import numpy as np

def load_data(file_path):
    # 加載數據
    data = np.load(file_path)
    positions = data['positions']
    forces = data['forces']
    return positions, forces

def preprocess_data(positions, forces):
    # 對數據進行標準化或其他預處理操作
    # 示例：將位置和力標準化到 [0, 1] 範圍
    positions = (positions - np.min(positions)) / (np.max(positions) - np.min(positions))
    forces = (forces - np.min(forces)) / (np.max(forces) - np.min(forces))
    return positions, forces
