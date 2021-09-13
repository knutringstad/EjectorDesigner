
def efficiencyPD(pandasSeries, fluid):

    mfr_m = pandasSeries["mfr_m"]
    mfr_s = pandasSeries["mfr_s"]
    ER= mfr_s/mfr_m
    Pm = pandasSeries["Pm"] 
    hm = pandasSeries["hm"]
    Ps = pandasSeries["Ps"] 
    hs = pandasSeries["hs"] 
    Po = pandasSeries["Po"]

    eff = efficiency(Pm, Po, Ps, hm, hs, ER, fluid)

    return eff


def efficiency(Pm, Po, Ps, hm, hs, ER, fluid):
    import CoolProp.CoolProp as CP

    ss = CP.PropsSI('S','P',Ps,'H',hs,fluid)
    sm = CP.PropsSI('S','P',Pm,'H',hm,fluid)
    hm_iso = CP.PropsSI('H','P',Po,'S',sm,fluid)
    hs_iso = CP.PropsSI('H','P',Po,'S',ss,fluid)

    eff = ER * (hs_iso - hs)/(hm - hm_iso)

    return eff

