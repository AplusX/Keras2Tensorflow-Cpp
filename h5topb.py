# convert .h5 to .pb
import tensorflow as tf
from tensorflow.python.framework import graph_io
from keras import backend as K
from nets.pspnet import pspnet
from keras.models import load_model
from keras.models import Model, load_model
from keras.models import model_from_json

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
   from tensorflow.python.framework.graph_util import convert_variables_to_constants
   graph = session.graph
   with graph.as_default():
       freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
       output_names = output_names or []
       output_names += [v.op.name for v in tf.global_variables()]
       input_graph_def = graph.as_graph_def()
       if clear_devices:
           for node in input_graph_def.node:
               node.device = ""
       frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
   return frozen_graph

K.set_learning_phase(0)


keras_model = load_model('./model.h5')
/*如果h5不包含模型结构，需要先load json然后再load weights
json_file = '/model.json'
with open(json_file, 'r') as f:
  json_str = f.read()
model = model_from_json(json_str)
keras_model.load_weights('./model.h5')
*/

# .inputs和.outputs非常重要，需要记录下来
print('Inputs are:', keras_model.inputs)
print('Outputs are:', keras_model.outputs)
// 保存pb文件
frozen_graph = freeze_session(K.get_session(), output_names=[keras_model.output.op.name])
graph_io.write_graph(frozen_graph, "./", "model.pb", as_text=False)

