from train_utils import DrivingData
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.io import output_notebook
import ipywidgets
import numpy as np
import bokeh.plotting as bkp
from bokeh.io import output_notebook, push_notebook
# 读取数据
path = ["/cailiu2/Diffusion-Planner/data/processed/us-nv-las-vegas-strip_7fc811bcf45f5e79.npz"]
output_notebook()

train_set = DrivingData(path, 10, 10)
ego, neighbors, _, _, _, _, _, _, _ = train_set[0]

# 确保数据维度正确
print("ego shape:", ego.shape)  # 应为 (num_frames, 2)
print("neighbors shape:", neighbors.shape)  # 应为 (num_neighbors, num_frames, 2)

num_frames = len(ego)
num_neighbors = len(neighbors)

# 提取数据
ego_x, ego_y = ego[:, 0], ego[:, 1]
neighbors_x, neighbors_y = neighbors[:, :, 0], neighbors[:, :, 1]

# 初始视图范围
initial_x, initial_y = ego_x[0], ego_y[0]
p = bkp.figure(
    title="Ego Vehicle & Neighbors",
    x_range=(initial_x - 50, initial_x + 50),
    y_range=(initial_y - 50, initial_y + 50),
    width=600, height=600
)

# 数据源
source = ColumnDataSource(data={
    'ego_x': [],
    'ego_y': [],
})

agents_source = ColumnDataSource(data={
    'neighbors_x': neighbors_x[:, 0].tolist(),
    'neighbors_y': neighbors_y[:, 0].tolist()
})

# 绘图
p.circle('ego_x', 'ego_y', source=source, size=12, color="red", legend_label="Ego Vehicle")
p.scatter('neighbors_x', 'neighbors_y', source=agents_source, size=6, color="blue", alpha=0.6, legend_label="Neighbors")

# 创建滑块类，初始化并绑定回调
slider_class = ipywidgets.IntSlider(
    value=0, min=0, max=num_frames-1, step=1, description="Frame"
)

# 滑块回调
def slider_callback(frame):
    print("frame: ", frame)
    new_ego_x = ego_x[frame]
    new_ego_y = ego_y[frame]
    print("new ego_x: ", new_ego_x)
    print("new ego_y: ", new_ego_y)

    # 更新数据
    source.data = {'ego_x': [new_ego_x], 'ego_y': [new_ego_y]}
    agents_source.data = {'neighbors_x': neighbors_x[:, frame].tolist(), 'neighbors_y': neighbors_y[:, frame].tolist()}

    p.circle('ego_x', 'ego_y', source=source, size=12, color="red", legend_label="Ego Vehicle")
    p.scatter('neighbors_x', 'neighbors_y', source=agents_source, size=6, color="blue", alpha=0.6, legend_label="Neighbors")
    bkp.show(p, notebook_handle=True)
    push_notebook()
# 显示图形



# 使用 `ipywidgets` 绑定回调
ipywidgets.interactive(slider_callback, frame=slider_class)
