"""Scientific interpretation shared by regenerated research metadata."""
ESTIMATION = ('Joint company and year fixed effects: outcome, regressors and year indicators '
              'are projected off company indicators. Company-clustered CR1 uses full-model '
              'residual degrees of freedom; p-values use t(G-1). Within R² includes year effects '
              'relative to the company-demeaned outcome. Plot intervals are normal-approximation '
              '95% intervals (coefficient +/- 1.96 SE).')

def align_metadata(a):
    a['analysisStructureBrief']['overview'] = (
        'Conditional associations between seven Business-topic disclosure measures and market '
        'capitalization in Japanese listed companies, 2016–2025. Relatedness is normalized '
        'semantic alignment, not absolute disclosure volume. M3 is an alternative specification '
        'and does not control for Relatedness. Fundamentals and analyst comparisons are '
        'descriptive sensitivity analyses; subgroup analyses are exploratory. Neither attenuation '
        'nor longer lags identify a causal mechanism. ' + ESTIMATION)
    aims = [
        'Construct exact calendar-year lags and annual market-capitalization changes.',
        'Compare separate topic associations under M1, M2 and alternative M3.',
        'Assess same-sample sensitivity to lagged ROA and leverage.',
        'Assess associations with analyst recommendations and same-sample adjustment.',
        'Describe signed positive and negative score coefficients.',
        'Explore associations across contemporaneous ROA quartiles.',
        'Assess alternative timing, outcomes and sample definitions.',
        'Explore topic associations within industries; no confirmed sector rankings.']
    notes = [
        'Missing prior calendar years produce missing lags; a prior row is not a prior year.',
        'Separate-topic regressions do not identify independent topic contributions. ' + ESTIMATION,
        'Coefficient attenuation is not a causal proportion and no arbitrary threshold establishes a channel.',
        'Stage 1 uses current recommendations; stage 2 controls lagged recommendations. Different timing and samples do not identify mediation. Recommendations measure assessment, not attention.',
        'Negative Mean is signed: a more negative value is a decrease. Pattern labels describe coefficient signs, not established mechanisms.',
        'Current ROA groups can change over time. Separate significance does not establish between-group differences.',
        'R0 is baseline plus five alternatives. Longer lags do not rule out reverse causality; growth is a different outcome.',
        'Exploratory comparisons involve multiple testing and some small company clusters. No adjusted p-values or formal between-sector contrasts are reported.']
    for i,lev in enumerate(a['analysisStructureBrief']['levels']):
        lev['levelAim']=aims[i];lev['interpretation']=notes[i]
        if i in [1,3,4]:lev['modelOrMethod'] = [
            '', ESTIMATION, '',
            'S1: relatedness(t-1) and recommendation(t); S2a: relatedness(t-1) and logMC(t); S2b adds recommendation(t-1) on the same S2 sample.',
            'M3 includes positive and signed negative mean scores, without Relatedness or Overall Sentiment.'
        ][i]
        lev['levelName']=lev['levelName'].replace('Mechanism Test','Descriptive Comparison').replace('Analyst Attention Channel','Analyst Recommendation Comparison').replace('Identification Robustness','Specification Sensitivity')
    for key,v in a['variableDictionary'].items():
        if key.startswith('match_'):
            v['variableDescription']='Normalized average semantic relatedness; matched to the same company in the previous calendar year. Not an absolute count of disclosure.'
    for key,v in a['figureDictionary'].items():
        v['figureCaption']=v['figureCaption'].replace('Coverage Volume','Relatedness').replace('Analyst Attention Channel (IBES Mediation)','Analyst Recommendation Comparison').replace('Identification Robustness','Specification Sensitivity')
        if key in ['fig06','fig07','fig08','fig09','fig10','fig11']:
            idx={'fig06':2,'fig07':3,'fig08':4,'fig09':5,'fig10':6,'fig11':7}[key]
            v['figureAim']=aims[idx];v['figureExplanation']=notes[idx]+' '+ESTIMATION
    for key,v in a['tableDictionary'].items():
        idx={'Table3':2,'Table4':3,'Table5':4,'Table6':5,'Table7':6,'Table8':7,'Table9':7}.get(key)
        if idx is not None:
            v['tableAim']=aims[idx];v['tableExplanation']=notes[idx]+' '+ESTIMATION
        v['tableCaption']=v['tableCaption'].replace('Mediation','Recommendation Comparison').replace('Mechanism','Descriptive').replace('Coverage Volume','Relatedness')
