
def MakeCase(Name):
    import os.path
    from os import path
    
    if os.path.isdir('./Cases/%s' %Name):
        print('Case exists ') 
    else:
        os.mkdir('./Cases/%s' %Name)
        os.mkdir('./Cases/%s/%s' %(Name,'Settings'))
        os.mkdir('./Cases/%s/%s' %(Name,'Results'))
        os.mkdir('./Cases/%s/%s' %(Name, 'Meshes_and_cases'))
        os.mkdir('./Cases/%s/%s' %(Name,'PreviousResults'))
        os.mkdir('./Cases/%s/%s/%s' %(Name,'Meshes_and_cases', 'JournalFiles'))
        os.mkdir('./Cases/%s/%s/%s' %(Name,'Meshes_and_cases', 'Meshes'))
        os.mkdir('./Cases/%s/%s/%s' %(Name,'Meshes_and_cases', 'Cases'))
        os.mkdir('./Cases/%s/%s/%s' %(Name,'Meshes_and_cases', 'MeshData'))
        os.mkdir('./Cases/%s/%s/%s' %(Name,'Results', 'FluentOutputs'))
        os.mkdir('./Cases/%s/%s/%s' %(Name,'Results', 'DataFiles'))


        from shutil import copyfile
        copyfile('./Cases/Default/Settings/CFDSettings.py', './Cases/%s/Settings/CFDSettings.py' %Name)
        copyfile('./Cases/Default/Settings/SamplingSettings.py', './Cases/%s/Settings/SamplingSettings.py' %Name)
        copyfile('./Cases/Default/Settings/DatabaseSettings.py', './Cases/%s/Settings/DatabaseSettings.py' %Name)
        copyfile('./Cases/Default/Settings/DataAnalysisSettings.py', './Cases/%s/Settings/DataAnalysisSettings.py' %Name)
        copyfile('./Cases/Default/Settings/BaselineConditions.py', './Cases/%s/Settings/BaselineConditions.py' %Name)
        copyfile('./Cases/Default/Settings/MeshingSettings.py', './Cases/%s/Settings/MeshingSettings.py' %Name)

        f=open('./Cases/%s/Settings/CaseSettings.py' %Name,'w')
        f.write('CurrentCaseName = \'%s\' \n' %Name) 
        f.write( 'MeshRoot = \'./Cases/%s/Meshes_and_cases\' %CurrentCaseName\n'  )
        f.write( 'ResultsRoot = \'./Cases/%s/Results\' %CurrentCaseName\n'  )
        f.write( 'SettingsRoot = \'./Cases/%s/Settings\' %CurrentCaseName\n'  )
        f.write( 'PreviousResultsRoot = \'./Cases/%s/PreviousResults\' %CurrentCaseName\n'  )
        f.close()


def LoadCase(Name):
    from shutil import copy
    print('Loading case %s' %Name)
    
    copy('./Cases/%s/Settings/CFDSettings.py' %Name, './CurrentSettings/CFDSettings.py'  )
    copy('./Cases/%s/Settings/CaseSettings.py' %Name, './CurrentSettings/CaseSettings.py'  )
    copy('./Cases/%s/Settings/SamplingSettings.py' %Name, './CurrentSettings/SamplingSettings.py'  )
    copy('./Cases/%s/Settings/DatabaseSettings.py' %Name, './CurrentSettings/DatabaseSettings.py'  )
    copy('./Cases/%s/Settings/DataAnalysisSettings.py' %Name, './CurrentSettings/DataAnalysisSettings.py'  )
    copy('./Cases/%s/Settings/BaselineConditions.py' %Name, './CurrentSettings/BaselineConditions.py'  )
    copy('./Cases/%s/Settings/MeshingSettings.py' %Name, './CurrentSettings/MeshingSettings.py'  )
    print('Updated all settings')






