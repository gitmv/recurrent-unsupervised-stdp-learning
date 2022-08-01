from PymoNNto.Exploration.AnalysisModules.Weight_Classifier_Pre import *
from PymoNNto.Exploration.AnalysisModules.Weight_Classifier_Post import *
from UI.Analysis_Modules.Backpropagation_Classifier import *
from UI.Analysis_Modules.Activity_Response_Classifier import *
from UI.Analysis_Modules.Activity_Response_Labeler import *
from UI.Analysis_Modules.Backpropagation_Labeler import *

def add_all_analysis_modules(ng):
    Weight_Classifier_Pre(ng)
    Weight_Classifier_Post(ng)
    Backpropagation_Classifier(ng)
    Activity_Response_Classifier(ng)
    Activity_Response_Labeler(ng)
    Backpropagation_Labeler(ng)
