from PymoNNto import *

class Labeler_Base(AnalysisModule):

    def initialize(self, neurons):
        self.add_tag('labeler')
        self.recon_matrices = {}

    def get_class_labels(self, label_tag, classes, text_generator):
        result = {}
        recon_matrices = self.get_class_recon_mats(label_tag, classes)
        for c in recon_matrices:
            result[c] = generate_text_from_recon_mat(recon_matrices[c], text_generator)
        return result

    def get_class_recon_mats(self, label_tag, classes):
        result = {}
        if label_tag in self.recon_matrices:
            label_matrices = self.recon_matrices[label_tag]
            if len(label_matrices) == len(classes):
                for c in unique(classes):
                    mask = (classes == c)
                    summed_recon_mat = np.sum(label_matrices[mask], axis=0)
                    result[c] = summed_recon_mat
            else:
                print('labels and classes have different sizes')
        else:
            print('label not found')
        return result


#print(np.sum(np.zeros((100, 20, 10)), axis=0).shape)

