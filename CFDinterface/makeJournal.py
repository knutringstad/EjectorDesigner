def makeJournal(ID,meshname,NIT):
    from CFDinterface.journalFileGenerator import journalFileHEM, journalFileDRHM
    import CurrentSettings.CFDSettings

    if CurrentSettings.CFDSettings.model =='HEM':
        return journalFileHEM(ID,meshname,NIT)

    