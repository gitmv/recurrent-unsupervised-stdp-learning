from PymoNNto import *

class Static_Classification(AnalysisModule):

    def initialize(self, neurons):
        self.add_tag('classifier')
        self.name = self.get_init_attr('name', super()._get_base_name_())
        self.classes = self.get_init_attr('classes', None)
        self.save_result(self.name, self.classes)

    def _get_base_name_(self):
        return self.name
