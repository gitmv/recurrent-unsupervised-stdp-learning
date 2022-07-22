from UI.Analysis_Modules.Labeler_Base import *

class Labeler_Backpropagation(Labeler_Base):

    def get_recon_mat_and_label(self, neurons, id):
        recon = compute_temporal_reconstruction(neurons.network, neurons, id, neurons.tags[0])
        text = generate_text_from_recon_mat(recon, neurons.network['Text_Generator', 0])
        return recon, text

    #def get_label(self, neurons, id):
    #    return self.get_recon_mat_and_label(neurons, id)[0]

    #def get_recon_mat(self, neurons, id):
    #    return self.get_recon_mat_and_label(neurons, id)[1]

    def execute(self, neurons, **kwargs):
        labels = []
        recon_matrices = []

        for i in range(neurons.size):
            self.update_progress(i/neurons.size*100)
            print('\rrecon: {}/{}'.format(i, neurons.size), end='')
            r, l = self.get_recon_mat_and_label(neurons, i)
            labels.append(l)
            recon_matrices.append(r)

        self.recon_matrices[self.current_key] = np.array(recon_matrices)

        return labels



