"""
Implement reduce_lr on plateau
"""
import tensorflow as tf


# Define the class
class ReduceLROnPlateau:
    """
    A class to reduce the learning rate on plateau and
    return a new optimizer with reduced lr rate
    """
    def __init__(self, learning_rate: float, patience: int, decay_rate: float, delta: float, min_lr: float, mode: str):
        """
        A class to reduce the learning rate on plateau

        Parameters
        ----------
        learning_rate : float
            The initial learning rate
        patience : int
            The amount of time to wait and see before decaying the lr.
        decay_rate : float
            The rate of decay. NOTE: lr = lr * decay_rate. Hence, mention decay_rate = 1E-1 or as such
        delta : float
            The minimum difference up-to which, we can consider it as improvement
        min_lr : float
            The minimum lr up-to which we can decay.
        mode : str, optional
            The mode to consider for the current metric
        """
        self.learning_rate = learning_rate
        self.patience = patience
        self.decay_rate = decay_rate
        self.monitor_var = 100.0 if mode == "min" else 0.0
        self.delta = delta
        self.min_lr = min_lr
        self.patience_count = 0
        self.mode = mode

    # Implement optimizer changing
    def _change_optimizer(self, optimizer: tf.keras.optimizers.Optimizer):

        if self.patience_count > self.patience:
            new_lr = self.learning_rate * self.decay_rate
            if new_lr >= self.min_lr:
                print("[WARN]: Reducing Learning Rate to {0}".format(new_lr))
                self.learning_rate = new_lr
                self.patience_count = 0
                optimizer.learning_rate = self.learning_rate
            else:
                print("[WARN]: Reduced to Min_LR Already")
        else:
            self.patience_count += 1

    def _max_mode(self, monitor_variable: float, optimizer: tf.keras.optimizers.Optimizer):

        if self.monitor_var > monitor_variable:

            # Difference
            diff = self.monitor_var - monitor_variable
            if diff > self.delta:
                self._change_optimizer(optimizer)
        else:
            self.patience_count = 0
            self.monitor_var = monitor_variable

            optimizer.learning_rate = self.learning_rate

    def _min_mode(self, monitor_variable: float, optimizer: tf.keras.optimizers.Optimizer):

        if self.monitor_var < monitor_variable:

            # Difference
            diff = monitor_variable - self.monitor_var
            if diff > self.delta:
                self._change_optimizer(optimizer)
        else:
            self.patience_count = 0
            self.monitor_var = monitor_variable

            optimizer.learning_rate = self.learning_rate

    def check_lr(self, monitor_variable: float, optimizer: tf.keras.optimizers.Optimizer):
        """
        Function to check the monitor variable whether its increasing or not.
        Parameters
        ----------
        monitor_variable : float
            The value to cross-check for the LR optimization

        optimizer : tf.keras.optimizers.Optimizer
            The keras optimizers to regularize the LR

        Returns
        ---------
        None
        """
        # Check to decay
        if self.mode == "max":
            self._max_mode(monitor_variable, optimizer)
        elif self.mode == "min":
            self._min_mode(monitor_variable, optimizer)
        else:
            raise NotImplementedError
        return None
