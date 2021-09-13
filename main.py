from Cases.CaseSetup import MakeCase, LoadCase
from Cases.StoreCaseResults import StoreCase
from Database_interface.interfaceDB import updateDataBase
from Database_interface.makeDatabase import makeDatabase
from ValidationData.Experiment_post_processing import pre_process_experiments
from DataAnalysis.FlowStructurePrediction.postprocessStructures import startPost

CaseName = 'NewCase'

# StoreCase(CaseName)

MakeCase(CaseName)
LoadCase(CaseName)

makeDatabase()
updateDataBase()

startPost()