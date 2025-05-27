"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        indices = np.arange(len(self.dataset))

        # Shuffle if required
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))

        # Calculate number of batches
        batch_size = self.batch_size
        num_samples = len(self.dataset)

        # Determine the number of batches to yield
        num_batches = num_samples // batch_size
        if not self.drop_last and num_samples % batch_size != 0:
            num_batches += 1

        # Yield batches
        for i in range(num_batches):
            # Get indices for current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)  # Handle last batch
            batch_indices = indices[start_idx:end_idx]

            # Initialize empty batch dictionary
            batch = {}

            # Build batch from individual samples
            for idx in batch_indices:
                sample = self.dataset[idx]

                # Initialize dict keys on first sample
                if not batch:
                    for k in sample.keys():
                        batch[k] = []

                # Add data from this sample
                for k, v in sample.items():
                    batch[k].append(v)

            # Convert lists to numpy arrays
            for k in batch:
                batch[k] = np.array(batch[k])

            yield batch

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last (self.drop_last)!                #
        ########################################################################

        num_samples = len(self.dataset)
        if self.drop_last:
            length = num_samples // self.batch_size
        else:
            length = (num_samples + self.batch_size - 1) // self.batch_size

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
