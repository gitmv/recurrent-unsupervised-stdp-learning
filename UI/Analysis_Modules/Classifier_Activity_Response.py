from UI.Analysis_Modules.Activity_Response_Behaviour import *

class Classifier_Activity_Response(AnalysisModule):

    def initialize(self, neurons):
        self.add_tag('classifier')

        if len(neurons['Activity_Response_Behaviour']) == 0:
            self.ARB_module = neurons.network.add_behaviours_to_object({1000: Activity_Response_Behaviour()}, neurons)[1000]
        else:
            self.ARB_module = neurons['Activity_Response_Behaviour', 0]


    def get_data_matrix(self, neurons):
        s = self.ARB_module.recon_data.shape
        data = self.ARB_module.recon_data.reshape((s[0], s[1] * s[2]))
        return data



