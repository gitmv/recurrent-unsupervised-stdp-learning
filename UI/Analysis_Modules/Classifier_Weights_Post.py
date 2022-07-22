from UI.Analysis_Modules.Classifier_base import *

class Classifier_Weights_Post(Classifier_base):

    def get_data_matrix(self, neurons):
        return get_partitioned_synapse_matrix(neurons, 'EE', 'W')
