from keras.models import load_model
from keras.utils import plot_model

# 加载已经训练好的模型
model = load_model('../DDPGmodel/lstm_model.h5')

# 绘制模型结构图并保存为文件
# plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True, dpi=96)
# 绘制模型结构图并保存为文件，设置水平排列
plot_model(model, to_file='lstm_network_horizontal.png', show_shapes=True, show_layer_names=True, expand_nested=True, rankdir='LR', dpi=300)
