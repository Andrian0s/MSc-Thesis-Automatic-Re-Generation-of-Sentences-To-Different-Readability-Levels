"""Implementation of different utility functions for adapter layers."""

import torch
import torch.nn as nn
import numpy as np

from transformers.activations import get_activation


class ReadabilityVectorsSetup:
    """
    A class for setting up readability vectors for tasks.
    """
    def __init__(self, adapter_config):
        """
        Initializes the ReadabilityVectorsSetup instance.
        
        Parameters:
        adapter_config (object): Configuration object containing task names, dimension of task embedding, and 
        readability_vector_style.
        """
        self.tasks = adapter_config.tasks
        self.readability_map = {'adv': int(adapter_config.task_embedding_dim / 3), 
                                'int': int(adapter_config.task_embedding_dim / 4), 
                                'ele': int(adapter_config.task_embedding_dim / 8 )} 
        self.n_task_embedding_dim = adapter_config.task_embedding_dim
        self.readability_vector_style = adapter_config.readability_vector_style
        self.task_to_readability_vector = {}

        # For each task, generate its readability vector
        for task_name in self.tasks:
            src_readability, tgt_readability = self.task_to_src_tgt_readability(task_name)
            self.task_to_readability_vector[task_name] = self.make_readability_hypernetwork_input_vector(src_readability, tgt_readability, self.readability_vector_style, self.readability_map, self.n_task_embedding_dim)
    
    def make_readability_hypernetwork_input_vector(self, src_readability, tgt_readability, readability_vector_style, readability_map, n_task_embedding_dim):
        """
        Creates a vector of the source and target readability scores, with half the size of n_task_embedding_dim for each pair of src_readability and tgt_readability.
        
        Parameters:
        src_readability (str): The readability level of the source text.
        tgt_readability (str): The readability level of the target text.
        readability_vector_style (str): Determines the format of the output readability vector 
                                        ('both', 'source_only', 'target_only', or 'difference').
        readability_map (dict): A dictionary mapping readability levels to their corresponding scores.
        n_task_embedding_dim (int): The size of the readability vector.

        Returns:
        numpy.ndarray: A numpy array containing the calculated readability vector.
        """
        half_embedding_dim = int(n_task_embedding_dim / 2)

        # Create the source and target readability vectors
        src_readability_vector = [1] * readability_map[src_readability] + [0] * (half_embedding_dim - readability_map[src_readability])
        tgt_readability_vector = [1] * readability_map[tgt_readability] + [0] * (half_embedding_dim - readability_map[tgt_readability])

        # Depending on the readability_vector_style, choose the appropriate output vector
        if readability_vector_style == 'both':
            readability_vector_np = np.array(src_readability_vector + tgt_readability_vector)
        elif readability_vector_style == 'source_only':
            readability_vector_np = np.array(src_readability_vector + [0] * half_embedding_dim)
        elif readability_vector_style == 'target_only':
            readability_vector_np = np.array(tgt_readability_vector + [0] * half_embedding_dim)
        elif readability_vector_style == 'difference':
            difference = readability_map[src_readability] - readability_map[tgt_readability]
            sign = -1 if difference > 0 else 1
            readability_vector = [sign] * abs(difference) + [0] * (n_task_embedding_dim - abs(difference))
            readability_vector_np = np.array(readability_vector)
        else: 
            raise ValueError('Invalid input_style: ' + readability_vector_style)
        return readability_vector_np
    
    def task_to_src_tgt_readability(self, task_name):
        """
        Extracts source and target readability from the given task name.

        Parameters:
        task_name (str): The task name from which to extract source and target readability.

        Returns:
        tuple: A tuple containing the source readability and target readability.
        """
        # Split the task name by underscore '_'
        split_name = task_name.split('_')
        if len(split_name) != 5:
            raise ValueError(f"Invalid task name: {task_name}")
        # The source readability is the fourth element, target readability is the fifth.
        src_readability, tgt_readability = split_name[3], split_name[4]
        return src_readability, tgt_readability

    def get_task_to_readability_vector(self):
        """
        Getter method for task_to_readability_vector attribute.

        Returns:
        dict: A dictionary with task names as keys and their corresponding readability vectors as values.
        """
        return self.task_to_readability_vector


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(config.task_embedding_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embeding_generator(task_embedding).view(-1)


class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.task_embedding_dim = config.projected_task_embedding_dim \
            if config.train_task_embeddings else config.task_embedding_dim
        self.weight_generator = linear_layer(self.task_embedding_dim, config.input_dim)
        self.bias_generator = linear_layer(self.task_embedding_dim, config.input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)


class TaskEmbeddingController(nn.Module):
    """
    Main module controlling task embeddings.
    """
    def __init__(self, config):
        """
        Initializes the TaskEmbeddingController with the given config.

        Parameters:
        config (object): An object containing the required configuration values like device, task_embedding_dim,
        tasks, task_to_embeddings, and train_task_embeddings.
        """
        super(TaskEmbeddingController, self).__init__()

        # Device configuration (CPU or GPU)
        self.device = config.device

        # The size of the task embeddings
        self.task_embedding_dim = config.task_embedding_dim

        self.random_initial_task_embeddings = config.random_initial_task_embeddings

        # List of task names
        self.tasks = config.tasks

        # A mapping from task names to their embeddings
        self.task_to_task_embeddings = {task: task for task in self.tasks}

        # If task embeddings are provided in config, use them
        if config.task_to_embeddings is not None:
            self.task_to_task_embeddings = config.task_to_embeddings
            self.tasks = self.task_to_task_embeddings.values()
        else:
            # Setting up the readability vectors for tasks
            self.task_to_readability_vector = ReadabilityVectorsSetup(config).get_task_to_readability_vector()

            # Initializing the task embeddings
            self.set_task_embeddings(self.tasks)

        # A flag indicating whether to train task embeddings or not
        self.train_task_embeddings = config.train_task_embeddings

        # If task embeddings are trainable, initialize the hypernetwork for task embeddings
        if self.train_task_embeddings:
            self.task_hyper_net = TaskHyperNet(config)
        
    def get_task(self, task):
        """
        Retrieves the task embedding for a given task.

        Parameters:
        task (str): The name of the task.

        Returns:
        torch.Tensor: The task embedding for the given task.
        """
        return self.task_to_task_embeddings[task]

    def set_task_embeddings(self, tasks):
        """
        Initializes the task embeddings for the given tasks.

        Parameters:
        tasks (list): A list of task names.
        """
        self.task_to_embeddings = nn.ParameterDict(dict())
        for task in tasks:
            # Converts the numpy array of the task's readability vector to a Torch tensor
            if self.random_initial_task_embeddings:
                task_embedding = torch.Tensor(torch.randn(self.task_embedding_dim)).to(self.device)
            else:
                task_embedding = torch.from_numpy(self.task_to_readability_vector[task]).float().to(self.device)
            self.task_to_embeddings[task] = nn.Parameter(task_embedding)

    def forward(self, task):
        """
        Computes the forward pass for a given task.

        Parameters:
        task (str): The name of the task.

        Returns:
        torch.Tensor: The output tensor after the forward pass.
        """
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_embeddings[task_mapped]
        
        # If task embeddings are trainable, pass them through the hypernetwork
        if self.train_task_embeddings:
            return self.task_hyper_net(task_embedding)
        return task_embedding
