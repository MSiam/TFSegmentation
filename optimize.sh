#First step save graph.pb by running in inference mode, with is_training as a boolean variable set to False not placeholder

#Note: for bazel, use TensorFlow source, and then install bazel.

#Second Run freeze graph to convert all variables to constant operations
python -m tensorflow.python.tools.freeze_graph --input_graph graph.pb --input_checkpoint experiments/coarsepre_fcn8s_shufflenet/checkpoints/best/-236744 --output_graph graph_frozen.pb --output_node_names=network/output/ArgMax
# OR For TF 1.7 from source
#bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=/home/mg/Desktop/Repositories/TFSegmentation/graph.pb --input_checkpoint=/home/mg/Desktop/Repositories/TFSegmentation/experiments/coarsepre_fcn8s_shufflenet/checkpoints/best/-236744 --output_graph=/home/mg/Desktop/Repositories/TFSegmentation/graph_frozen.pb --output_node_names=network/output/ArgMax



#Third call optimize_for_inference for batchnorm folding and merging operations (This is sometimes buggy)
python -m tensorflow.python.tools.optimize_for_inference --input graph_frozen.pb --output graph_optimized.pb --input_names=network/input/Placeholder --output_names=network/output/ArgMax
# OR For TF 1.7 from source
#If not built, build it from source.
#bazel build tensorflow/tools/graph_transforms:transform_graph

#bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=/home/mg/Desktop/Repositories/TFSegmentation/graph_frozen.pb --out_graph=/home/mg/Desktop/Repositories/TFSegmentation/graph_optimized.pb --inputs='import/network/input/Placeholder:0' --inputs='import/network/input/Placeholder_2:0' --outputs='network/output/ArgMax:0' --transforms='
#  strip_unused_nodes()
#  remove_nodes(op=CheckNumerics)
#  fold_constants(ignore_errors=true)
#  fold_batch_norms
#  fold_old_batch_norms'
