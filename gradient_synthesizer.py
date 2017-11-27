import tensorflow as tf
import numpy as np

class SynthGradBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l1=1.0, l2=0.0):
        op_name = "SynGrad%d" % self.num_calls

        @tf.RegisterGradient(op_name)
        def _grad_synth(op, grad):
            PDy, PDd = tf.unstack(grad)
            shape_info = len(PDy.shape)

            if shape_info > 1:
                dim = 1
            else:
                dim = 0

            PDy_n = tf.nn.l2_normalize(PDy, dim=dim)
            PDd_n = tf.nn.l2_normalize(PDd, dim=dim)

            if shape_info > 1:
                contraction_dim = np.arange(shape_info-1)+1
            else:
                contraction_dim = 0

            scale = l1 * tf.reduce_sum(PDy_n * PDd_n, axis=contraction_dim) + l2

            reshape_shape = np.ones(shape_info, dtype=np.int)
            reshape_shape[0] = -1

            scale_v = tf.reshape(scale, reshape_shape)
            grad_bp = PDy + scale_v * PDd
            return [grad_bp, tf.zeros_like(PDd)]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Pack": op_name}):
            y = tf.stack([x, x])

        self.num_calls += 1
        return y

GradSys = SynthGradBuilder()