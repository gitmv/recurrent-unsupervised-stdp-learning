from UI.Analysis_Modules.Labeler_Base import *
from UI.Analysis_Modules.Activity_Response_Behaviour import *

class Labeler_Activity_Response(Labeler_Base):

    def initialize(self, neurons):
        super().initialize(neurons)

        if len(neurons['Activity_Response_Behaviour']) == 0:
            self.ARB_module = neurons.network.add_behaviours_to_object({1000: Activity_Response_Behaviour()}, neurons)[1000]
        else:
            self.ARB_module = neurons['Activity_Response_Behaviour', 0]

    def execute(self, neurons, **kwargs):

        self.recon_matrices[self.current_key] = self.ARB_module.recon_data.copy()

        labels = []
        sequence_indx = np.argmax(self.ARB_module.recon_data, axis=2)
        alphabet = np.array(self.ARB_module.generator.alphabet)
        chars = alphabet[sequence_indx]

        for n in range(self.ARB_module.neuron_size):
            text = ''.join(chars[n])
            labels.append(text.replace(' ', '_'))

        return labels



    #def get_data_matrix(self, neurons):
    #    s = self.ARB_module.recon_data.shape
    #    data = self.ARB_module.recon_data.reshape((s[0], s[1] * s[2]))
    #    return data

