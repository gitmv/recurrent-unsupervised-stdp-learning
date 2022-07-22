from PymoNNto import *

class Classifier_Backpropagation(AnalysisModule):

    def initialize(self, neurons):
        self.add_tag('classifier')

    def get_data_matrix(self, neurons):
        module = neurons['Labeler_Backpropagation', 0]
        if module is not None:

            if module.last_call_result() is None:
                print('generating labels for classification')
                neurons['Labeler_Backpropagation', 0].exec()

            data = module.last_call_result()  # [labels,matrices]
            if data is not None:
                data = data[1]
                s = data.shape
                return data.reshape((s[0], s[1] * s[2]))
            else:
                print('Cannot find Labeler_Backpropagation results.')
        else:
            print('Cannot find Labeler_Backpropagation module. Make sure that it is attached to same group.')
