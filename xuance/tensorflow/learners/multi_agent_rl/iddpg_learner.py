"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import Sequence
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 optimizer: Sequence[tk.optimizers.Optimizer],
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.sync_frequency = sync_frequency
        super(IDDPG_Learner, self).__init__(config, policy, optimizer, device, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'])
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            rewards = tf.convert_to_tensor(sample['rewards'])
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], dtype=tf.float32), [-1, self.n_agents, 1])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                    [-1, self.n_agents, 1])
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
            batch_size = obs.shape[0]

            # train actor
            with tf.GradientTape() as tape:
                # calculate the loss function
                inputs = {"obs": obs, "ids": IDs}
                _, actions_eval = self.policy(inputs)
                loss_a = -tf.reduce_sum(self.policy.critic(obs, actions_eval, IDs) * agent_mask) / tf.reduce_sum(agent_mask)
                gradients = tape.gradient(loss_a, self.policy.parameters_actor)
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_actor)
                    if grad is not None
                ])

            # train critic
            with tf.GradientTape() as tape:
                inputs_next = {"obs": obs_next, "ids": IDs}
                q_eval = self.policy.critic(obs, actions, IDs)
                q_next = self.policy.target_critic(obs_next, self.policy.target_actor(inputs_next), IDs)
                q_target = rewards + (1 - terminals) * self.args.gamma * q_next
                y_pred = tf.reshape(q_eval * agent_mask, [-1])
                y_true = tf.stop_gradient(tf.reshape(q_target * agent_mask, [-1]))
                loss_c = tk.losses.mean_squared_error(y_true, y_pred)
                gradients = tape.gradient(loss_c, self.policy.parameters_critic)
                self.optimizer['critic'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_critic)
                    if grad is not None
                ])

            self.policy.soft_update(self.tau)

            lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
            lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

            info = {
                "learning_rate_actor": lr_a.numpy(),
                "learning_rate_critic": lr_c.numpy(),
                "loss_actor": loss_a.numpy(),
                "loss_critic": loss_c.numpy(),
                "predictQ": tf.math.reduce_mean(q_eval).numpy()
            }

            return info
