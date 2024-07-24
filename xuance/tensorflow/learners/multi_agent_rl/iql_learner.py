"""
Independent Q-learning (IQL)
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(IQL_Learner, self).__init__(config, policy, optimizer, device, model_dir)

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            rewards = tf.convert_to_tensor(sample['rewards'])
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], dtype=tf.float32), [-1, self.n_agents, 1])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32), [-1, self.n_agents, 1])
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
            batch_size = obs.shape[0]

            with tf.GradientTape() as tape:
                inputs_policy = {"obs": obs, "ids": IDs}
                _, _, q_eval = self.policy(inputs_policy)
                q_eval_a = tf.gather(q_eval, tf.reshape(actions, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                inputs_target = {"obs": obs_next, "ids": IDs}
                q_next = self.policy.target_Q(inputs_target)

                if self.args.double_q:
                    _, action_next_greedy, q_next_eval = self.policy(inputs_target)
                    action_next_greedy = tf.reshape(tf.cast(action_next_greedy, dtype=tf.int64), [batch_size, self.n_agents, 1])
                    q_next_a = tf.gather(q_next, action_next_greedy, axis=-1, batch_dims=-1)
                else:
                    q_next_a = tf.reduce_max(q_next, axis=-1, keepdims=True)

                q_target = rewards + (1-terminals) * self.args.gamma * q_next_a

                # calculate the loss function
                q_eval_a *= agent_mask
                q_target *= agent_mask
                q_target = tf.stop_gradient(tf.reshape(q_target, [-1]))
                q_eval_a = tf.reshape(q_eval_a, [-1])
                loss = tk.losses.mean_squared_error(q_target, q_eval_a)
                gradients = tape.gradient(loss, self.policy.trainable_param())
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_param())
                    if grad is not None
                ])

            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "learning_rate": lr.numpy(),
                "loss_Q": loss.numpy(),
                "predictQ": tf.math.reduce_mean(q_eval_a).numpy()
            }

            return info
