from PymoNNto import *

class Activity_Response_Behaviour(Behaviour):

    def set_variables(self, neurons):
        self.steps = self.get_init_attr('steps', 10)
        self.generator = neurons.network['Text_Generator', 0]
        self.alp_size = len(self.generator.alphabet)
        self.decay = self.get_init_attr('decay', 0.99999)
        self.neuron_size = neurons.size
        self.recon_data = np.zeros((self.neuron_size, self.steps, self.alp_size))


    def new_iteration(self, neurons):
        text = self.generator.history[-self.steps:]
        o = neurons.output > 0

        for i, t in enumerate(text):
            indx = self.generator.char_to_index(t)
            self.recon_data[o, i, indx] += 1

        self.recon_data *= self.decay

    def get_reconstruction_matrix(self, indx):
        return self.recon_data[indx]



    def get_neuron_similarity(self, indx):
        s = np.sum(np.sum(self.recon_data - self.recon_data[indx], axis=2), axis=1)
        ms = np.max(s)
        if ms == 0:
            ms = 1
        res = (ms-s)/ms
        return res

    def get_representation(self, indx):
        sequence_indx = np.argmax(self.recon_data[indx], axis=1)
        alphabet = np.array(self.generator.alphabet)
        chars = alphabet[sequence_indx]
        return ''.join(chars).replace(' ', '_')
