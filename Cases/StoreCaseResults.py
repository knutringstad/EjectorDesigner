
def StoreCase(CaseName):

    # If you want to use same case, but want to store results in a folder
    from Cases.CaseSetup import LoadCase
    

    LoadCase(CaseName)
    import CurrentSettings.CaseSettings

    SaveName = 'Saved_old_results'

    from distutils.dir_util import copy_tree
    copy_tree(CurrentSettings.CaseSettings.MeshRoot, CurrentSettings.CaseSettings.PreviousResultsRoot + '/' + SaveName +'/Meshes_and_cases')
    copy_tree(CurrentSettings.CaseSettings.ResultsRoot, CurrentSettings.CaseSettings.PreviousResultsRoot + '/' + SaveName +'/Results')

    print('Results in case %s successfully stored in subfolder %s ' %(CaseName,SaveName))